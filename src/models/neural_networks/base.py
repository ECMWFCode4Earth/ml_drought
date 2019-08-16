import numpy as np
import random
from pathlib import Path
import pickle
import math

import torch
from torch.nn import functional as F

import shap

from typing import Dict, List, Optional, Tuple

from ..base import ModelBase
from ..utils import chunk_array
from ..data import DataLoader, train_val_mask, TrainData, idx_to_input


class NNBase(ModelBase):

    def __init__(self,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 include_latlons: bool = False,
                 include_monthly_aggs: bool = True,
                 include_yearly_aggs: bool = True,
                 surrounding_pixels: Optional[int] = None,
                 ignore_vars: Optional[List[str]] = None,
                 include_static: bool = True,
                 device: str = 'cuda:0') -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months, include_pred_month,
                         include_latlons, include_monthly_aggs, include_yearly_aggs,
                         surrounding_pixels, ignore_vars, include_static)

        # for reproducibility
        if (device != 'cpu') and torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'
        torch.manual_seed(42)

        self.explainer: Optional[shap.DeepExplainer] = None

    def to(self, device: str = 'cpu'):
        # move the model onto the right device
        raise NotImplementedError

    def explain(self, x: Optional[List[torch.Tensor]] = None,
                var_names: Optional[List[str]] = None,
                save_shap_values: bool = True) -> Dict[str, np.ndarray]:
        """
        Expain the outputs of a trained model.

        Arguments
        ----------
        x: The values to explain. If None, samples are randomly drawn from
            the test data
        var_names: The variable names of the historical inputs. If x is None, this
            will be calculated. Only necessary if the arrays are going to be saved
        save_shap_values: Whether or not to save the shap values

        Returns
        ----------
        shap_dict: A dictionary of shap values for each of the model's input arrays
        """
        assert self.model is not None, 'Model must be trained!'

        if self.explainer is None:
            background_samples = self._get_background(sample_size=100)
            self.explainer: shap.DeepExplainer = shap.DeepExplainer(
                self.model, background_samples)

        if x is None:
            # if no input is passed to explain, take 10 values and explain them
            test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=1,
                                            shuffle_data=True, mode='test', to_tensor=True,
                                            static=True, experiment=self.experiment)
            key, val = list(next(iter(test_arrays_loader)).items())[0]
            x = self.make_shap_input(val.x, start_idx=0, num_inputs=10)
            var_names = val.x_vars

        explain_arrays = self.explainer.shap_values(x)

        if save_shap_values:
            analysis_folder = self.model_dir / 'analysis'
            if not analysis_folder.exists():
                analysis_folder.mkdir()
            for idx, shap_array in enumerate(explain_arrays):
                np.save(analysis_folder / f'shap_value_{idx_to_input[idx]}.npy', shap_array)
                np.save(analysis_folder / f'input_{idx_to_input[idx]}.npy', x[idx].cpu().numpy())

            # save the variable names too
            if var_names is not None:
                with (analysis_folder / 'input_variable_names.pkl').open('wb') as f:
                    pickle.dump(var_names, f)

        return {idx_to_input[idx]: array for idx, array in enumerate(explain_arrays)}

    def _initialize_model(self, x_ref: Tuple[torch.Tensor, ...]) -> torch.nn.Module:
        raise NotImplementedError

    def train(self, num_epochs: int = 1,
              early_stopping: Optional[int] = None,
              batch_size: int = 256,
              learning_rate: float = 1e-3,
              val_split: float = 0.1) -> None:
        print(f'Training {self.model_name} for experiment {self.experiment}')

        if early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     experiment=self.experiment,
                                                     shuffle_data=False,
                                                     pred_months=self.pred_months))
            train_mask, val_mask = train_val_mask(len_mask, val_split)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train',
                                          experiment=self.experiment,
                                          mask=train_mask,
                                          to_tensor=True,
                                          pred_months=self.pred_months,
                                          ignore_vars=self.ignore_vars,
                                          monthly_aggs=self.include_monthly_aggs,
                                          surrounding_pixels=self.surrounding_pixels,
                                          static=self.include_static,
                                          device=self.device)

            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='train',
                                        experiment=self.experiment,
                                        mask=val_mask,
                                        to_tensor=True,
                                        pred_months=self.pred_months,
                                        ignore_vars=self.ignore_vars,
                                        monthly_aggs=self.include_monthly_aggs,
                                        surrounding_pixels=self.surrounding_pixels,
                                        static=self.include_static,
                                        device=self.device)

            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train',
                                          experiment=self.experiment,
                                          to_tensor=True,
                                          pred_months=self.pred_months,
                                          ignore_vars=self.ignore_vars,
                                          monthly_aggs=self.include_monthly_aggs,
                                          surrounding_pixels=self.surrounding_pixels,
                                          static=self.include_static,
                                          device=self.device)

        # initialize the model
        if self.model is None:
            x_ref, _ = next(iter(train_dataloader))
            model = self._initialize_model(x_ref)
            self.model = model

        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=learning_rate)

        for epoch in range(num_epochs):
            train_rmse = []
            train_l1 = []
            self.model.train()
            for x, y in train_dataloader:
                for x_batch, y_batch in chunk_array(x, y, batch_size, shuffle=True):
                    optimizer.zero_grad()
                    pred = self.model(x_batch[0],
                                      self._one_hot_months(x_batch[1]),  # type: ignore
                                      x_batch[2],
                                      x_batch[3],
                                      x_batch[4],
                                      x_batch[5])
                    loss = F.smooth_l1_loss(pred, y_batch)
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
                        val_pred_y = self.model(x[0],
                                                self._one_hot_months(x[1]),
                                                x[2],
                                                x[3],
                                                x[4],
                                                x[5])
                        val_loss = F.mse_loss(val_pred_y, y)

                        val_rmse.append(math.sqrt(val_loss.cpu().item()))

            print(f'Epoch {epoch + 1}, train smooth L1: {np.mean(train_l1)}, '
                  f'RMSE: {np.mean(train_rmse)}')

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f'Val RMSE: {epoch_val_rmse}')
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_model_dict = self.model.state_dict()
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print('Early stopping!')
                        self.model.load_state_dict(best_model_dict)
                        return None

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='test',
                                        experiment=self.experiment,
                                        pred_months=self.pred_months, to_tensor=True,
                                        ignore_vars=self.ignore_vars,
                                        monthly_aggs=self.include_monthly_aggs,
                                        surrounding_pixels=self.surrounding_pixels,
                                        static=self.include_static,
                                        device=self.device)

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert self.model is not None, 'Model must be trained before predictions can be generated'

        self.model.eval()
        with torch.no_grad():
            for dict in test_arrays_loader:
                for key, val in dict.items():
                    preds = self.model(
                        val.x.historical, self._one_hot_months(val.x.pred_months),
                        val.x.latlons, val.x.current, val.x.yearly_aggs, val.x.static
                    )
                    preds_dict[key] = preds.cpu().numpy()
                    test_arrays_dict[key] = {'y': val.y.cpu().numpy(),
                                             'latlons': val.latlons,
                                             'time': val.target_time}

        return test_arrays_dict, preds_dict

    def _get_background(self,
                        sample_size: int = 150) -> List[torch.Tensor]:

        print('Extracting a sample of the training data')

        train_dataloader = DataLoader(data_path=self.data_path,
                                      batch_file_size=self.batch_size,
                                      shuffle_data=True, mode='train',
                                      pred_months=self.pred_months,
                                      to_tensor=True,
                                      ignore_vars=self.ignore_vars,
                                      monthly_aggs=self.include_monthly_aggs,
                                      surrounding_pixels=self.surrounding_pixels,
                                      static=self.include_static,
                                      device=self.device)

        output_tensors: List[torch.Tensor] = []
        output_pm: List[torch.Tensor] = []
        output_ll: List[torch.Tensor] = []
        output_cur: List[torch.Tensor] = []
        output_ym: List[torch.Tensor] = []
        output_static: List[torch.Tensor] = []

        samples_per_instance = max(1, sample_size // len(train_dataloader))

        for x, _ in train_dataloader:
            while len(output_tensors) < sample_size:
                for _ in range(samples_per_instance):
                    idx = random.randint(0, x[0].shape[0] - 1)
                    output_tensors.append(x[0][idx])

                    # one hot months
                    one_hot_months = self._one_hot_months(x[1][idx: idx + 1])
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
                    if x[5] is None:
                        output_static.append(torch.zeros(1))
                    else:
                        output_static.append(x[5][idx])

        return [torch.stack(output_tensors),  # type: ignore
                torch.cat(output_pm, dim=0),
                torch.stack(output_ll),
                torch.stack(output_cur),
                torch.stack(output_ym),
                torch.stack(output_static)]

    @staticmethod
    def _one_hot_months(indices: torch.Tensor) -> torch.Tensor:
        return torch.eye(14)[indices.long()][:, 1:-1]

    def make_shap_input(self, x: TrainData, start_idx: int = 0,
                        num_inputs: int = 10) -> List[torch.Tensor]:
        """
        Returns a list of tensors, as is required
        by the shap explainer
        """
        output_tensors = []
        output_tensors.append(x.historical[start_idx: start_idx + num_inputs])
        # one hot months
        one_hot_months = self._one_hot_months(x.pred_months[start_idx: start_idx + num_inputs])
        output_tensors.append(one_hot_months[start_idx: start_idx + num_inputs])
        output_tensors.append(x.latlons[start_idx: start_idx + num_inputs])
        if x.current is None:
            output_tensors.append(torch.zeros(num_inputs, 1))
        else:
            output_tensors.append(x.current[start_idx: start_idx + num_inputs])
        # yearly aggs
        output_tensors.append(x.yearly_aggs[start_idx: start_idx + num_inputs])
        # static data
        if x.static is None:
            output_tensors.append(torch.zeros(num_inputs, 1))
        else:
            output_tensors.append(x.static[start_idx: start_idx + num_inputs])
        return output_tensors
