import numpy as np
import random
from pathlib import Path
import pickle
import math
import xarray as xr

import torch
from torch.nn import functional as F

from typing import cast, Dict, List, Optional, Tuple, Union

from ..base import ModelBase
from ..utils import chunk_array, _to_xarray_dataset
from ..data import DataLoader, train_val_mask, TrainData, idx_to_input

shap = None


class NNBase(ModelBase):
    def __init__(
        self,
        data_folder: Path = Path("data"),
        batch_size: int = 1,
        experiment: str = "one_month_forecast",
        pred_months: Optional[List[int]] = None,
        include_pred_month: bool = True,
        include_latlons: bool = False,
        include_monthly_aggs: bool = True,
        include_yearly_aggs: bool = False,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        device: str = "cuda:0",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
        clear_nans: bool = True,
        weight_observations: bool = False,
        explain: bool = False,
    ) -> None:
        super().__init__(
            data_folder=data_folder,
            batch_size=batch_size,
            experiment=experiment,
            pred_months=pred_months,
            include_pred_month=include_pred_month,
            include_latlons=include_latlons,
            include_monthly_aggs=include_monthly_aggs,
            include_yearly_aggs=include_yearly_aggs,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_prev_y=include_prev_y,
            normalize_y=normalize_y,
            clear_nans=clear_nans,
        )

        # for reproducibility
        if (device != "cpu") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        torch.manual_seed(42)

        if explain:
            global shap
            if shap is None:
                import shap
            self.explainer: Optional[shap.DeepExplainer] = None  # type: ignore
        self.weight_observations = weight_observations

    def to(self, device: str = "cpu"):
        # move the model onto the right device
        raise NotImplementedError

    def _make_analysis_folder(self) -> Path:
        analysis_folder = self.model_dir / "analysis"
        if not analysis_folder.exists():
            analysis_folder.mkdir()
        return analysis_folder

    def _initialize_model(self, x_ref: Tuple[torch.Tensor, ...]) -> torch.nn.Module:
        raise NotImplementedError

    def _make_weights(
        self,
        target: torch.Tensor,
        threshold_value: float = 15,
        weight_value: float = 10,
    ) -> torch.Tensor:
        """weight the gradient updates to learn more from certain
        observations!

        Returns a vector of the same size as target.
        The values are weight else 1

        E.g. upweight the low VCI examples so the model pays
        greater attention to these layers.

        Arguments:
        ---------
        target: torch.Tensor
            The y variable (the regression target)
        """
        # if normalize y then use -1 STD otherwise use
        # the extreme droughts (<= threshold value)
        # threshold_value = -1 if self.normalize_y else threshold_value
        # weight = 10
        weights = torch.ones_like(target)
        weights[target <= threshold_value] = weight_value

        return weights

    def train(
        self,
        num_epochs: int = 1,
        early_stopping: Optional[int] = None,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        val_split: float = 0.1,
        check_inversion: bool = False,
    ) -> None:
        print(f"Training {self.model_name} for experiment {self.experiment}")

        if early_stopping is not None:
            len_mask = len(
                DataLoader._load_datasets(
                    self.data_path,
                    mode="train",
                    experiment=self.experiment,
                    shuffle_data=False,
                    pred_months=self.pred_months,
                )
            )
            train_mask, val_mask = train_val_mask(len_mask, val_split)

            train_dataloader = self.get_dataloader(
                mode="train", mask=train_mask, to_tensor=True, shuffle_data=True
            )
            val_dataloader = self.get_dataloader(
                mode="train", mask=val_mask, to_tensor=True, shuffle_data=False
            )

            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = self.get_dataloader(
                mode="train", to_tensor=True, shuffle_data=True
            )

        # initialize the model
        if self.model is None:
            x_ref, _ = next(iter(train_dataloader))
            model = self._initialize_model(x_ref)
            self.model = model

        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()], lr=learning_rate
        )

        for epoch in range(num_epochs):
            train_rmse = []
            train_l1 = []
            self.model.train()
            for x, y in train_dataloader:
                for x_batch, y_batch in chunk_array(x, y, batch_size, shuffle=True):
                    optimizer.zero_grad()
                    pred = self.model(
                        *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
                    )
                    if (epoch == 0) & check_inversion:  # check only the first epoch
                        # create xarray objects
                        pred_xr = _to_xarray_dataset(latlons=x_batch[2], data=pred)
                        true_xr = _to_xarray_dataset(latlons=x_batch[2], data=y_batch)
                        # check that nans more or less the same
                        assert (
                            pred_xr.isnull().data.values == true_xr.isnull().data.values
                        ).mean() > 0.92, (
                            "The missing data should be the same for 92% of the data. "
                            "This sometimes occurs when there has been a problem with an inversion "
                            "somewhere in the data"
                        )

                    loss = F.smooth_l1_loss(pred, y_batch)

                    # upweight the losses on the extreme deficits
                    if self.weight_observations:
                        weights = self._make_weights(
                            y_batch, threshold_value=15, weight_value=10
                        )

                        loss = (weights * loss).mean()

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        rmse = F.mse_loss(pred, y_batch)
                        train_rmse.append(math.sqrt(rmse.cpu().item()))

                    train_l1.append(loss.item())

            if early_stopping is not None:
                self.model.eval()
                val_rmse = []
                with torch.no_grad():
                    for x, y in val_dataloader:
                        val_pred_y = self.model(*self._input_to_tuple(x))
                        val_loss = F.mse_loss(val_pred_y, y)

                        val_rmse.append(math.sqrt(val_loss.cpu().item()))

            print(
                f"Epoch {epoch + 1}, train smooth L1: {np.mean(train_l1)}, "
                f"RMSE: {np.mean(train_rmse)}"
            )

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f"Val RMSE: {epoch_val_rmse}")
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_model_dict = self.model.state_dict()
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print("Early stopping!")
                        self.model.load_state_dict(best_model_dict)
                        return None

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = self.get_dataloader(
            mode="test", to_tensor=True, shuffle_data=False
        )

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert (
            self.model is not None
        ), "Model must be trained before predictions can be generated"

        self.model.eval()
        with torch.no_grad():
            for dict in test_arrays_loader:
                for key, val in dict.items():

                    input_tuple = self._input_to_tuple(val.x)

                    # TODO - this code is mostly copied from
                    # models.utils - can be cleaned up
                    # with a default batch size of 256
                    num_sections = max(input_tuple[0].shape[0] // 256, 1)
                    split_x = []
                    for idx, x_section in enumerate(input_tuple):
                        if x_section is not None:
                            split_x.append(torch.chunk(x_section, num_sections))
                        else:
                            split_x.append([None] * num_sections)  # type: ignore

                    chunked_input = list(zip(*split_x))

                    all_preds = []
                    for batch in chunked_input:
                        all_preds.append(self.model(*batch).cpu().numpy())
                    preds_dict[key] = np.concatenate(all_preds)

                    test_arrays_dict[key] = {
                        "y": val.y.cpu().numpy(),
                        "latlons": val.latlons,
                        "time": val.target_time,
                        "y_var": val.y_var,
                    }
                    if self.predict_delta:
                        assert val.historical_target.shape[0] == val.y.shape[0], (
                            "Expect"
                            f"the shape of the y ({val.y.shape})"
                            f" and historical_target ({val.historical_target.shape})"
                            " to be the same!"
                        )
                        test_arrays_dict[key][
                            "historical_target"
                        ] = val.historical_target

        return test_arrays_dict, preds_dict

    def _get_background(
        self, sample_size: int = 150
    ) -> List[Union[torch.Tensor, None]]:

        print("Extracting a sample of the training data")

        train_dataloader = self.get_dataloader(
            mode="train", shuffle_data=True, to_tensor=True
        )

        output_tensors: List[torch.Tensor] = []
        output_pm: List[torch.Tensor] = []
        output_ll: List[torch.Tensor] = []
        output_cur: List[torch.Tensor] = []
        output_ym: Optional[List[torch.Tensor]] = None
        output_static: List[torch.Tensor] = []
        output_prev_y: List[torch.Tensor] = []

        samples_per_instance = max(1, sample_size // len(train_dataloader))

        for x, _ in train_dataloader:
            for _ in range(samples_per_instance):
                idx = random.randint(0, x[0].shape[0] - 1)
                output_tensors.append(x[0][idx])

                # one hot months
                one_hot_months = self._one_hot(x[1][idx : idx + 1], 12)
                output_pm.append(one_hot_months)

                # latlons
                output_ll.append(x[2][idx])

                # current array
                if x[3] is None:
                    output_cur.append(torch.zeros(1))
                else:
                    output_cur.append(x[3][idx])

                # yearly aggs
                if x[4] is None:
                    if output_ym is None:
                        output_ym = [torch.zeros(1)]
                    else:
                        output_ym.append(torch.zeros(1))
                else:
                    output_ym.append(x[4][idx])  # type: ignore

                # static data
                if self.static == "embeddings":
                    output_static.append(
                        self._one_hot(x[5][idx], cast(int, self.num_locations))
                    )
                elif self.static == "features":
                    output_static.append(x[5][idx])
                else:
                    output_static.append(torch.zeros(1))

                output_prev_y.append(x[6][idx])

                if len(output_tensors) >= sample_size:
                    return [
                        torch.stack(output_tensors).to(self.device),  # type: ignore
                        torch.cat(output_pm, dim=0).to(self.device),
                        torch.stack(output_ll).to(self.device),
                        torch.stack(output_cur).to(self.device),
                        torch.stack(output_ym).to(self.device)
                        if output_ym is not None
                        else None,
                        torch.stack(output_static).to(self.device),
                        torch.stack(output_prev_y).to(self.device),
                    ]

        return [
            torch.stack(output_tensors),  # type: ignore
            torch.cat(output_pm, dim=0),
            torch.stack(output_ll),
            torch.stack(output_cur),
            torch.stack(output_ym) if output_ym is not None else None,
            torch.stack(output_static),
            torch.stack(output_prev_y),
        ]

    def _one_hot(self, indices: torch.Tensor, num_vals: int) -> torch.Tensor:
        if len(indices.shape) > 1:
            indices = indices.squeeze(-1)
        return torch.eye(num_vals + 2, device=self.device)[indices.long()][:, 1:-1]

    def _input_to_tuple(
        self, x: Union[Tuple[torch.Tensor, ...], TrainData]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
        --------
        Tuple:
            [0] historical data
            [1] months (one hot encoded)
            [2] latlons
            [3] current data
            [4] yearly aggregations
            [5] static data
            [6] prev y var
        """
        # mypy totally fails to handle what's going on here

        if type(x) is TrainData:  # type: ignore
            return (  # type: ignore
                x.historical,  # type: ignore
                self._one_hot(x.pred_months, 12),  # type: ignore
                x.latlons,  # type: ignore
                x.current,  # type: ignore
                x.yearly_aggs,  # type: ignore
                self._one_hot(x.static, self.num_locations)  # type: ignore
                if self.static == "embeddings"
                else x.static,  # type: ignore
                x.prev_y_var,  # type: ignore
            )
        else:
            return (
                x[0],  # type: ignore
                self._one_hot(x[1], 12),  # type: ignore
                x[2],  # type: ignore
                x[3],  # type: ignore
                x[4],  # type: ignore
                self._one_hot(x[5], self.num_locations)  # type: ignore
                if self.static == "embeddings"
                else x[5],  # type: ignore
                x[6],  # type: ignore
            )

    def explain(
        self,
        x: Optional[TrainData] = None,
        var_names: Optional[List[str]] = None,
        save_explanations: bool = True,
        background_size: int = 100,
        start_idx: int = 0,
        num_inputs: int = 10,
        method: str = "shap",
    ) -> Tuple[TrainData, List[Union[torch.Tensor, None]]]:
        """
        Expain the outputs of a trained model.

        Arguments
        ----------
        x: The values to explain. If None, samples are randomly drawn from
            the test data
        var_names: The variable names of the historical inputs. If x is None, this
            will be calculated. Only necessary if the arrays are going to be saved
        background_size: the size of the background to use
        save_shap_values: Whether or not to save the shap values

        Returns
        ----------
        shap_dict: A dictionary of shap values for each of the model's input arrays
        """

        assert self.model is not None, "Model must be trained!"
        if x is None:
            # if no input is passed to explain, take 10 values and explain them
            test_arrays_loader = self.get_dataloader(
                mode="test", shuffle=False, batch_file_size=1, to_tensor=True
            )
            _, val = list(next(iter(test_arrays_loader)).items())[0]
            var_names = val.var_names
            x = val.x

        if method == "shap":
            explanations, background_samples = self._get_shap_explanations(
                x, background_size, start_idx, num_inputs
            )
        elif method == "morris":
            explanations = self._get_morris_explanations(x)
            background_samples = None  # type: ignore

        if save_explanations:
            analysis_folder = self._make_analysis_folder()
            for idx, expl_array in enumerate(explanations):
                org_array = x.__getattribute__(idx_to_input[idx])
                if org_array is not None:
                    np.save(
                        analysis_folder / f"{method}_value_{idx_to_input[idx]}.npy",
                        expl_array,
                    )
                    np.save(
                        analysis_folder / f"{method}_{idx_to_input[idx]}.npy",
                        org_array.detach().cpu().numpy(),
                    )

            # save the variable names too
            if var_names is not None:
                with (analysis_folder / "input_variable_names.pkl").open("wb") as f:
                    pickle.dump(var_names, f)

        return TrainData(**explanations), background_samples

    def _get_shap_explanations(
        self,
        x: TrainData,
        background_size: int = 100,
        start_idx: int = 0,
        num_inputs: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], List[Union[torch.Tensor, None]]]:

        if self.explainer is None:  # type: ignore
            background_samples = self._get_background(sample_size=background_size)
            self.explainer: shap.DeepExplainer = shap.DeepExplainer(  # type: ignore
                self.model, background_samples
            )

        # make val.x a list of tensors, as is required by the shap explainer
        output_tensors = []
        for _, val in sorted(idx_to_input.items()):
            tensor = x.__getattribute__(val)
            if tensor is not None:
                if val == "pred_months":
                    output_tensors.append(
                        self._one_hot(tensor[start_idx : start_idx + num_inputs], 12)
                    )
                elif val == "static":
                    if self.static == "embeddings":
                        assert x.static is not None
                        output_tensors.append(
                            self._one_hot(
                                x.static[start_idx : start_idx + num_inputs],
                                cast(int, self.num_locations),
                            )
                        )
                    else:
                        assert x.static is not None
                        output_tensors.append(
                            x.static[start_idx : start_idx + num_inputs]
                        )
                else:
                    output_tensors.append(tensor[start_idx : start_idx + num_inputs])
            else:
                output_tensors.append(torch.zeros(num_inputs, 1))

        explain_arrays = self.explainer.shap_values(output_tensors)  # type: ignore

        return (
            {idx_to_input[idx]: array for idx, array in enumerate(explain_arrays)},
            background_samples,  #  None
        )

    def _get_morris_explanations(self, x: TrainData) -> Dict[str, np.ndarray]:
        """
        https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/morris.py

        Will return a train data object with the Morris gradients of the inputs
        """

        self.model.eval()
        self.model.zero_grad()

        for idx, (key, val) in enumerate(x.__dict__.items()):
            if val is not None:
                val.requires_grad = True
        outputs = self.model(
            x.historical,
            self._one_hot(x.pred_months, 12),
            x.latlons,
            x.current,
            x.yearly_aggs,
            x.static,
            x.prev_y_var,
        )

        num_items = len(x.__dict__)
        output_dict: Dict[str, Optional[np.ndarray]] = {}
        for idx, (key, val) in enumerate(x.__dict__.items()):
            if val is not None:
                grad = torch.autograd.grad(
                    outputs,
                    val,
                    retain_graph=True if idx + 1 < num_items else None,
                    allow_unused=True,
                    grad_outputs=torch.ones_like(outputs).to(self.device),
                )[0]
                if grad is not None:
                    # this can be the case since allow_unused = True
                    output_dict[key] = grad.detach().cpu().numpy()
                else:
                    output_dict[key] = None
            else:
                output_dict[key] = None

        return output_dict
