import numpy as np
import random
from pathlib import Path
import pickle
import xarray as xr
import tqdm

import torch
from torch.nn import functional as F

import shap

from typing import cast, Dict, List, Optional, Tuple, Union

from .nseloss import NSELoss
from ..base import ModelBase
from ..utils import chunk_array
from ..data import (
    DataLoader,
    train_val_mask,
    TrainData,
    idx_to_input,
    timestamp_train_val_mask,
)
from ..dynamic_data import DynamicDataLoader


class NNBase(ModelBase):
    def __init__(
        self,
        data_folder: Path = Path("data"),
        dynamic: bool = False,
        batch_size: int = 256,
        experiment: str = "one_month_forecast",
        forecast_horizon: int = 1,
        target_var: Optional[str] = None,
        test_years: Optional[Union[List[str], str]] = None,
        seq_length: int = 365,
        pred_months: Optional[List[int]] = None,
        include_pred_month: bool = True,
        include_latlons: bool = False,
        include_timestep_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        dynamic_ignore_vars: Optional[List[str]] = None,
        static_ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        device: str = "cuda:0",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
        val_years: Optional[List[Union[float, int]]] = None,
        train_years: Optional[List[Union[float, int]]] = None,
        clip_values_to_zero: bool = False,
    ) -> None:
        super().__init__(
            dynamic=dynamic,
            data_folder=data_folder,
            batch_size=batch_size,
            experiment=experiment,
            seq_length=seq_length,
            include_pred_month=include_pred_month,
            include_latlons=include_latlons,
            include_timestep_aggs=include_timestep_aggs,
            include_yearly_aggs=include_yearly_aggs,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            static=static,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_prev_y=include_prev_y,
            normalize_y=normalize_y,
            pred_months=pred_months,
            dynamic_ignore_vars=dynamic_ignore_vars,
            static_ignore_vars=static_ignore_vars,
            target_var=target_var,
            test_years=test_years,
            forecast_horizon=forecast_horizon,
            clip_values_to_zero=clip_values_to_zero,
        )

        # for reproducibility
        if (device != "cpu") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        torch.manual_seed(42)

        self.explainer: Optional[shap.DeepExplainer] = None

        self.train_years = train_years
        self.val_years = val_years

        if self.train_years is not None:
            assert not any(
                np.isin(test_years, train_years)
            ), "MODEL LEAKAGE - Train > Test"
        if self.val_years is not None:
            assert not any(np.isin(test_years, val_years)), "MODEL LEAKAGE - Val > Test"

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

    def _val_epoch(
        self,
        val_rmses: List[float],
        val_dataloader: Union[DataLoader, DynamicDataLoader],
    ) -> List[float]:
        self.model.eval()
        val_batch_rmse = []

        with torch.no_grad():
            for x, y in val_dataloader:
                val_pred_y = self.model(*self._input_to_tuple(x))
                val_loss = F.mse_loss(val_pred_y, y)

                # check for nan loss
                assert not np.isnan(val_loss.cpu().item())

                # validation loss
                val_batch_rmse.append(np.sqrt(val_loss.cpu().item()))

        # assert val_batch_rmse != []
        val_rmses.append(np.mean(val_batch_rmse))

        return val_rmses

    def _train_epoch(
        self,
        epoch: int,
        train_dataloader: Union[DataLoader, DynamicDataLoader],
        train_rmse: List[float],
        train_losses: List[float],
        learning_rate: Union[float, Dict[int, float]],
        loss_func: str,
    ) -> Union[List[float], List[float]]:
        """Run epoch training step"""
        # Adaptive Learning Rate
        if isinstance(learning_rate, dict):
            if epoch in learning_rate.keys():
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate[epoch]

        epoch_rmses = []
        epoch_losses = []

        # ----------------------------------------
        # Training
        # ----------------------------------------
        self.model.train()
        # load in a few timesteps at a time (sample xy by TIME)
        for x, y in tqdm.tqdm(train_dataloader):
            # chunk into n_pixels (BATCHES)
            for x_batch, y_batch in chunk_array(x, y, self.batch_size, shuffle=True):
                self.optimizer.zero_grad()

                # ------- FORWARD PASS ---------
                pred = self.model(
                    *self._input_to_tuple(cast(Tuple[torch.Tensor, ...], x_batch))
                )
                # ------- LOSS FUNCTION ---------
                if loss_func == "NSE":
                    # NSELoss needs std of each basin for each sample
                    if x_batch[7] is not None:
                        target_var_std = x_batch[7]
                        # (train_dataloader.__iter__()).target_var_std
                    else:
                        assert (
                            False
                        ), "x[7] should not be None, this is the target_var_std"

                    loss = NSELoss().forward(pred, y_batch, target_var_std)
                elif loss_func == "MSE":
                    loss = F.mse_loss(pred, y_batch)
                elif loss_func == "huber":
                    loss = F.smooth_l1_loss(pred, y_batch)
                else:
                    assert False, "Only implemented MSE / NSE / huber loss functions"

                # ------- BACKWARD PASS ---------
                loss.backward()
                self.optimizer.step()

                # evaluation / keeping track of losses over time
                with torch.no_grad():
                    rmse = F.mse_loss(pred, y_batch)

                epoch_losses.append(loss.cpu().item())
                epoch_rmses.append(np.sqrt(rmse.cpu().item()))

                assert len(epoch_losses) >= 1
                assert len(epoch_rmses) >= 1

                # TODO: check that most recent loss is notnan
                assert not np.isnan(epoch_losses[-1])

        # update the lists of mean epoch loss
        train_rmse.append(np.mean(epoch_rmses))
        train_losses.append(np.mean(epoch_losses))

        # check the losses are not nans
        # TODO: the final batch is always nan - why?
        assert not np.isnan(np.nanmean(epoch_losses))
        assert not np.isnan(np.nanmean(epoch_rmses))

        # Print the losses to the user
        print(
            f"Epoch {epoch + 1}, train loss: {np.mean(epoch_losses)}, "
            f"RMSE: {np.mean(train_rmse)}"
        )

        return train_rmse, train_losses

    def _init_train_val_periods(
        self, dataloader: Union[DataLoader, DynamicDataLoader]
    ) -> Tuple[List[bool], List[bool]]:
        """Return a boolean list of the train and validation periods"""
        # randomly selected
        if (self.val_years is None) and (self.train_years is None):
            len_mask = len(dataloader.valid_train_times)
            train_mask, val_mask = train_val_mask(len_mask, 0.1)

        # Provided by the user
        else:
            all_times = dataloader.valid_train_times
            # TODO: select the validation period
            train_mask, val_mask = timestamp_train_val_mask(
                all_times=all_times,
                train_years=self.val_years,
                val_years=self.val_years,
            )
        return train_mask, val_mask

    def train(
        self,
        num_epochs: int = 1,
        early_stopping: Optional[int] = None,
        learning_rate: Union[Dict[int, float], float] = 1e-3,
        val_split: float = 0.1,
        loss_func: str = "MSE",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"Training {self.model_name} for experiment {self.experiment}")

        assert loss_func in [
            "MSE",
            "NSE",
            "huber",
        ], f"loss_func must be one of: ['MSE', 'NSE', 'huber'] \nGot {loss_func}"

        # ----------------------------------------
        # Initialize the DataLoaders
        # ----------------------------------------
        if early_stopping is not None:
            if self.dynamic:
                dl = self.get_dataloader(mode="train")

                # get the train / validation period
                train_mask, val_mask = self._init_train_val_periods(dl)

                print("\n** Loading Dataloaders ... **")
                train_dataloader = self.get_dataloader(
                    mode="train", mask=train_mask, to_tensor=True, shuffle_data=True
                )
                val_dataloader = self.get_dataloader(
                    mode="train", mask=val_mask, to_tensor=True, shuffle_data=False
                )
                batches_without_improvement = 0
                best_val_score = np.inf

                # check correct length of the train/validation periods
                assert len(val_dataloader.valid_train_times) == np.sum(val_mask)
                assert len(train_dataloader.valid_train_times) == np.sum(train_mask)
            else:
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

                print("\n** Loading Dataloaders ... **")
                train_dataloader = self.get_dataloader(
                    mode="train", mask=train_mask, to_tensor=True, shuffle_data=True
                )
                val_dataloader = self.get_dataloader(
                    mode="train", mask=val_mask, to_tensor=True, shuffle_data=False
                )

                batches_without_improvement = 0
                best_val_score = np.inf
        else:
            print("\n** Loading Dataloaders ... **")
            train_dataloader = self.get_dataloader(
                mode="train", to_tensor=True, shuffle_data=True
            )

        # ----------------------------------------
        # Initialize the Model & Optimizer
        # ----------------------------------------
        print("\n** Initializing Model ... **")
        if self.model is None:
            x_ref, _ = next(iter(train_dataloader))
            model = self._initialize_model(x_ref)
            self.model = model

        self.optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=learning_rate if isinstance(learning_rate, float) else learning_rate[0],
        )

        print("\n** Running Epochs ... **")
        train_rmse = []
        train_losses = []
        val_rmses = []

        for epoch in range(num_epochs):
            train_rmse, train_losses = self._train_epoch(
                epoch=epoch,
                train_dataloader=train_dataloader,
                train_rmse=train_rmse,
                train_losses=train_losses,
                learning_rate=learning_rate,
                loss_func=loss_func
            )

            # ----------------------------------------
            # Validation
            # ----------------------------------------
            # epoch - check the accuracy on the validation set
            if early_stopping is not None:
                val_rmses = self._val_epoch(
                    val_rmses=val_rmses, val_dataloader=val_dataloader
                )

                # TODO: why are there nan validation scores?
                epoch_val_rmse = np.nanmean(val_rmses)
                assert not np.isnan(epoch_val_rmse)

                # do we want to stop training?
                print(f"Val RMSE: {epoch_val_rmse}")
                best_model_dict = self.model.state_dict()

                # new best score
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_model_dict = self.model.state_dict()
                else:  # early stopping
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print("Early stopping!")
                        self.model.load_state_dict(best_model_dict)
                        return (train_rmse, train_losses, val_rmses)

                val_rmses.append(epoch_val_rmse)

        return (train_rmse, train_losses, val_rmses)

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        print(f"** Making Predictions for {self.model_name} **")

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
            for dict in tqdm.tqdm(test_arrays_loader):
                for key, val in tqdm.tqdm(dict.items()):

                    # TODO: this is where the code breaks down
                    # ipdb> self.x.historical.shape => (659, 365, 8)
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
                        "id_to_loc_map": val.id_to_loc_map,
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

    def _get_background(self, sample_size: int = 150) -> List[torch.Tensor]:

        print("Extracting a sample of the training data")

        train_dataloader = self.get_dataloader(
            mode="train", shuffle_data=True, to_tensor=True
        )

        output_tensors: List[torch.Tensor] = []
        output_pm: List[torch.Tensor] = []
        output_ll: List[torch.Tensor] = []
        output_cur: List[torch.Tensor] = []
        output_ym: List[torch.Tensor] = []
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
                output_ym.append(x[4][idx])

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
                        torch.stack(output_tensors),  # type: ignore
                        torch.cat(output_pm, dim=0),
                        torch.stack(output_ll),
                        torch.stack(output_cur),
                        torch.stack(output_ym),
                        torch.stack(output_static),
                        torch.stack(output_prev_y),
                    ]

        return [
            torch.stack(output_tensors),  # type: ignore
            torch.cat(output_pm, dim=0),
            torch.stack(output_ll),
            torch.stack(output_cur),
            torch.stack(output_ym),
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
            [2] latlons (optional)
            [3] current data (optional - Nowcast)
            [4] yearly aggregations
            [5] static data (optional)
            [6] prev y var (optional)
            [7] y var std (optional - req for NSE loss function)
        """
        # mypy totally fails to handle what's going on here

        if type(x) is TrainData:  # type: ignore
            return (  # type: ignore
                x.historical,  # type: ignore
                self._one_hot(x.pred_month, 12)  # type: ignore
                if x.pred_month is not None  # type: ignore
                else None,
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
                self._one_hot(x[1], 12) if x[1] is not None else None,  # type: ignore
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
    ) -> TrainData:
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
            explanations = self._get_shap_explanations(
                x, background_size, start_idx, num_inputs
            )
        elif method == "morris":
            explanations = self._get_morris_explanations(x)

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

        return TrainData(**explanations)

    def _get_shap_explanations(
        self,
        x: TrainData,
        background_size: int = 100,
        start_idx: int = 0,
        num_inputs: int = 10,
    ) -> Dict[str, np.ndarray]:

        if self.explainer is None:
            background_samples = self._get_background(sample_size=background_size)
            self.explainer: shap.DeepExplainer = shap.DeepExplainer(  # type: ignore
                self.model, background_samples
            )

        # make val.x a list of tensors, as is required by the shap explainer
        output_tensors = []

        for _, val in sorted(idx_to_input.items()):
            tensor = x.__getattribute__(val)
            if tensor is not None:
                if val == "seq_length":
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
                output_tensors.append(torch.zeros(num_inputs, 1))

        explain_arrays = self.explainer.shap_values(output_tensors)

        return {idx_to_input[idx]: array for idx, array in enumerate(explain_arrays)}

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
            self._one_hot(x.pred_month, self.seq_length),
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

    def move_model(self, new_device: str) -> None:
        """Move the model between devices
            (e.g. 'cuda:0' -> 'cpu'
        """
        self.model.to(torch.device(new_device))
        self.device = new_device
