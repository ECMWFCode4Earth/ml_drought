from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
import itertools
import pandas as pd
import numpy as np
import xarray as xr
import json
import matplotlib.pyplot as plt

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer
import calendar


def highlight_cell(x, y, ax=None, fill=False, **kwargs):
    """Pick a particular cell to add a patch around
    https://stackoverflow.com/a/56655069/9940782
    """
    rect = plt.Rectangle((x - 0.5, y - 0.575), 1, 1, fill=fill, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def highlight_timestep_cells(
    ax, highlight_timesteps: List[pd.Timestamp], highlight_color: Optional[str] = None
):
    yr_index_map = {
        int(float(label.get_text())): ix
        for ix, label in enumerate(ax.get_yticklabels())
    }

    # get the indexes of the yr-mth
    yrs = [yr_index_map[dt.year] for dt in highlight_timesteps]
    mths = [dt.month - 1 for dt in highlight_timesteps]
    highlight_ixs = list(zip(mths, yrs))
    # highlight them all!
    highlight_color = "red" if highlight_color is None else highlight_color
    for cell in highlight_ixs:
        highlight_cell(cell[0], cell[1], ax=ax, color=highlight_color, fill=False)

    return ax


def make_monthly_calendar_plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    highlight_timesteps: Optional[List[pd.Timestamp]] = None,
    highlight_color: Optional[str] = None,
    **kwargs,
):
    """Make a calendar plot (months on the x axis, years on the y axis)
    from a DataFrame.

    Note:
    - The dataframe must have a time index
        ```
        df.astype(columns={'time': pd.Timestamp}).set_index('time')
        ```
    - the max/min years should be appended to the dataset
        ```
        _df = pd.DataFrame({value_column: [np.nan]}, index=[pd.to_datetime(MIN_TIME)])
        df = pd.concat([_df, df])
        _df = pd.DataFrame({value_column: [np.nan]}, index=[pd.to_datetime(MAX_TIME)])
        df = pd.concat([df, _df])
        ```
    - the dataset should be resampled to ensure that all empty
    months are filled with np.nan using:
        ```df = df.resample('M').first() ```
    - must assign a "year" and a "month" column to the dataframe
        ```
        df['month'] = [pd.to_datetime(d).month for d in df.index]
        df['year'] = [pd.to_datetime(d).year for d in df.index]
        ```
    """
    assert "year" in [c for c in df.columns]
    assert "month" in [c for c in df.columns]

    try:
        im = ax.imshow(
            df.pivot(index="year", columns="month").values, aspect="auto", **kwargs
        )
    except ValueError as E:
        print(E)
        im = ax.imshow(
            df.reset_index().drop("index").pivot(index="year", columns="month").values,
            aspect="auto",
            **kwargs,
        )

    ax.set_xticks([i for i in range(0, 12)])
    ax.set_xticklabels([calendar.month_abbr[i + 1] for i in range(0, 12)])
    ax.set_xlabel("Month")
    plt.xticks(rotation=45)

    ax.set_yticks([i for i in range(len(df.year.unique()))])
    ax.set_yticklabels([yr for yr in range(df.year.min(), df.year.max() + 1)])
    ax.set_ylabel("Year")

    ax.set_title(title)

    if highlight_timesteps is not None:
        ax = highlight_timestep_cells(
            ax=plt.gca(),
            highlight_timesteps=highlight_timesteps,
            highlight_color=highlight_color,
        )

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(14)

    fig = plt.gcf()
    cbar = fig.colorbar(im)
    cbar.set_label(
        [c for c in df.columns if c not in ["year", "month"]][0], fontsize=14
    )

    return ax


def calculate_length_of_hi_med_lo_experiment_train_years(
    total_months: int, test_length: int = 12, pred_timesteps: int = 3
) -> List[int]:
    # how many months do we have for train/test
    total_train_months = total_months - (test_length * (pred_timesteps + 1))

    # split into 3 groups
    # round up the group sizes
    train_length_1 = train_length_2 = round(total_train_months / 3)
    train_length_0 = round(total_train_months / 3)

    if not train_length_0 + train_length_1 + train_length_2 == total_train_months:
        train_length_0 = round(total_train_months / 3) + 1

    assert train_length_0 + train_length_1 + train_length_2 == total_train_months, (
        "" f"{train_length_0 + train_length_1 + train_length_2} == {total_train_months}"
    )

    # calculate the number of train months in each experiment
    train_lengths = [
        train_length_0,
        train_length_0 + train_length_1,
        train_length_0 + train_length_1 + train_length_2,
    ]

    return train_lengths


def get_valid_test_timesteps(
    pred_timesteps: int, target_data: pd.DataFrame
) -> pd.DataFrame:
    """Need at least `pred_timesteps + 1` before the test timestep to
    allow enough previous timesteps of predictor variables!
    """
    return target_data.iloc[pred_timesteps + 1 :, :]


def sort_by_median_target_var(
    target_data: xr.Dataset,
    pred_timesteps: int,
    data_dir: Path = Path("data"),
    drop_nans: bool = True,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """ Calculate the sorted_timesteps to then calculate hi/med/lo
    train and test periods.
    """
    target_variable = [v for v in target_data.data_vars][0]

    # SORT BY MEDIAN VCI EACH MONTH (over space)
    median_data = target_data.resample(time="M").mean().median(dim=["lat", "lon"])
    median_data = median_data.to_dataframe()
    median_data = get_valid_test_timesteps(
        pred_timesteps=pred_timesteps, target_data=median_data
    )

    # sorted low to high
    sorted_df = median_data.sort_values(target_variable)
    if drop_nans:
        sorted_df = sorted_df.dropna()
    sorted_timesteps = sorted_df.index

    return sorted_df, sorted_timesteps


class Experiment:
    """
    train_length: int
        the length of the training period (# timesteps)
    test_length: int
        the length of the testing period (# timesteps)
    train_hilo: str
        selecting the training years from which tercile?
        one of ['high', 'med', 'low']
    test_hilo: str
        selecting the training years from which tercile?
        one of ['high', 'med', 'low']
    sorted_timesteps: pd.DatetimeIndex
        The timesteps sorted by some function
        Usually, using the `sort_by_median_target_var` function
    TODO: put the get_experiment_years function inside this class!
    """

    def __init__(
        self,
        train_length: int,
        train_hilo: str,
        test_hilo: str,
        test_length: int,
        sorted_timesteps: pd.DatetimeIndex,
        pred_timesteps: int = 3,
        randomly_select_test_times: bool = True,
    ):
        self.train_length = train_length
        self.train_hilo = train_hilo
        self.test_hilo = test_hilo
        self.test_length = test_length
        self.sorted_timesteps = sorted_timesteps
        self.pred_timesteps = pred_timesteps
        self.randomly_select_test_times = randomly_select_test_times

        assert train_hilo in ["high", "med", "low"]
        assert test_hilo in ["high", "med", "low"]

        # NOTE: do we reshuffle or take from the already prescribed groups?
        # e.g. do we recalculate the hi/med/lo groups after we have
        # selected the test years?
        self.reshuffle = True

        # NOTE: do we sample with replacement?
        # e.g. do we give the model the opportunity to retrain on the same years?
        self.replacement = False

        test_timesteps, train_timesteps = self.get_experiment_timesteps(
            sorted_timesteps, randomly_select_test_times=self.randomly_select_test_times
        )
        self.test_timesteps = [pd.to_datetime(ts) for ts in test_timesteps]
        self.train_timesteps = [pd.to_datetime(ts) for ts in train_timesteps]

        # CHECK NO DATA LEAKAGE
        assert ~all(np.isin(self.test_timesteps, self.train_timesteps)), (
            f"Data Leakage:\n\nTrain timesteps: {[f'{ts.year}-{ts.month}' for ts in train_timesteps]}\n\n"
            "Test Timesteps: {[f'{ts.year}-{ts.month}' for ts in train_timesteps]}"
        )

    @staticmethod
    def _calculate_hilo_dict(sorted_timesteps: np.array) -> Dict[str, np.array]:
        # split into three groups
        low, med, high = tuple(np.array_split(sorted_timesteps, 3))
        hilo_dict = {"low": low, "med": med, "high": high}

        return hilo_dict

    def get_test_timesteps_plus(self, test_timesteps: np.array) -> pd.DatetimeIndex:
        """ENSURE selecting train timesteps NOT including the test timesteps
            as predictor timesteps. I.e. to prevent Data Leakage (train->test)
        """
        all_timesteps = self.sorted_timesteps.copy().sort_values()

        dict_ = {
            f"test_{ix}": bool_arr
            for ix, bool_arr in enumerate([all_timesteps == t for t in test_timesteps])
        }
        df = pd.DataFrame(dict_)

        # get the index values for the TEST months + pred_timesteps
        # because these timesteps cannot be seen by the TRAIN data
        # to prevent model leakage
        list_of_invalid_indexes = [
            [
                i
                for i in range(
                    df.index[df[col]][0],
                    (df.index[df[col]] + (self.pred_timesteps + 1))[0],
                )
            ]
            for col in df.columns
        ]

        list_of_invalid_indexes = np.array(list_of_invalid_indexes).flatten()
        bool_invalid_list = [
            True if i in list_of_invalid_indexes else False
            for i, ts in enumerate(all_timesteps)
        ]
        test_timesteps_plus = all_timesteps[bool_invalid_list]

        return test_timesteps_plus

    @staticmethod
    def get_n_middle(seq, n_elements) -> np.array:
        """Get the n_elements from the middle of a sequence"""
        # get the middle index
        idx = len(seq) // 2
        # get n above/below
        n_below = n_elements // 2
        n_above = (n_elements // 2) + (n_elements % 2)
        assert n_below + n_above == n_elements

        # extract the middle indices from sequence
        middle_portion = np.append(seq[idx - n_below : idx], seq[idx : idx + n_above])
        assert len(middle_portion) == n_elements

        return middle_portion

    def get_experiment_timesteps(
        self, sorted_timesteps: np.array, randomly_select_test_times: bool = True
    ) -> Tuple[np.array, np.array]:
        # 1. split into low, med, high groups
        test_dict = self._calculate_hilo_dict(sorted_timesteps)

        # 2. select randomly the TEST timesteps from these groups
        if randomly_select_test_times:
            test_timesteps = np.random.choice(
                test_dict[self.test_hilo], self.test_length, replace=False
            )
        else:  # OR select the middle, or extremes (hilo)
            if self.test_hilo == "med":
                mid_idx = int(len(sorted_timesteps) / 2)
                # select 12 middle elements
                test_timesteps = self.get_n_middle(sorted_timesteps, 12)
            elif self.test_hilo == "high":
                # select 12 highest
                test_timesteps = sorted_timesteps[-12:]
            elif self.test_hilo == "low":
                # select 12 lowest
                test_timesteps = sorted_timesteps[:12]

        # 3. remove test_timesteps from array to choose train_timesteps
        # we also need to account for the fact that these TEST timesteps
        #  should not be seen by the models (e.g. as predictor timesteps)
        test_timesteps_plus = self.get_test_timesteps_plus(test_timesteps)
        if self.reshuffle:
            sorted_timesteps_for_training = np.array(
                [ts for ts in self.sorted_timesteps if ts not in test_timesteps_plus]
            )
            train_dict = self._calculate_hilo_dict(sorted_timesteps_for_training)

        else:
            train_dict = test_dict.copy()
            train_dict[self.train_hilo] = np.array(
                [
                    ts
                    for ts in train_dict[self.train_hilo]
                    if ts not in test_timesteps_plus
                ]
            )

        # 4. Choose the train_timesteps
        # get the n training timesteps from the group
        # NOTE: PREVENT DATA LEAKAGE (no data used as input to the models)
        if self.train_length > len(train_dict[self.train_hilo]):
            # if the number of training timesteps is more than the group
            # we need to 'steal' some timesteps from other groups
            train_timesteps = train_dict[self.train_hilo]

            leftover_timesteps = [
                i
                for i in sorted_timesteps
                if (i not in train_timesteps) & (i not in test_timesteps_plus)
            ]

            # iteratively split the array into low, med, high and keep selecting
            # test_timesteps_plus from the correct group until they're all used up
            while len(train_timesteps) < self.train_length:
                # a. no more timesteps left to select
                if len(leftover_timesteps) == 0:
                    print("No more timesteps left for training")
                    break

                # calculate a new lo,med,hi dictionary
                train_dict = self._calculate_hilo_dict(leftover_timesteps)

                # if we run out of timesteps in our chosen group
                # e.g. {low=[1991], med=[1992], high=[]}
                # and we want high training timesteps, select from other groups
                if len(train_dict[self.train_hilo]) == 0:
                    print(
                        f"not enough {self.train_hilo} timesteps left! Selecting from other groups"
                    )
                    train_timesteps = np.append(
                        train_timesteps, np.random.choice(leftover_timesteps, 1)
                    )
                    leftover_timesteps = [
                        i
                        for i in self.sorted_timesteps
                        if (i not in train_timesteps) & (i not in test_timesteps_plus)
                    ]

                    # append remaining to the train_timesteps
                    train_timesteps = np.append(train_timesteps, leftover_timesteps)

                    continue
                train_timesteps = np.append(
                    train_timesteps, train_dict[self.train_hilo]
                )
                leftover_timesteps = [
                    i
                    for i in sorted_timesteps
                    if (i not in train_timesteps) & (i not in test_timesteps_plus)
                ]

        else:
            train_timesteps = np.random.choice(
                train_dict[self.train_hilo], self.train_length, replace=self.replacement
            )

        return test_timesteps, train_timesteps

    def plot_experiment_split(
        self, ax: Optional = None, show_test_timesteps: bool = True
    ):
        # FILL IN GAPS!
        is_test = []
        for ts in self.sorted_timesteps:
            if ts in self.test_timesteps:
                is_test.append(1)
            elif ts in self.train_timesteps:
                is_test.append(0)
            else:
                is_test.append(np.nan)

        df = pd.DataFrame(
            {"Test Data": is_test}, index=self.sorted_timesteps
        ).sort_index()

        df = df.sort_index()
        df["year"] = df.index.year
        df["month"] = df.index.month

        if ax is None:
            fig, ax = plt.subplots()

        title = f"Test: {self.test_hilo} // Train: {self.train_hilo}\nTrainLength:{self.train_length} TestLength:{self.test_length}"
        if show_test_timesteps:
            highlight_timesteps = [pd.to_datetime(dt) for dt in self.test_timesteps]
        else:
            highlight_timesteps = None
        ax = make_monthly_calendar_plot(
            df, ax, title=title, highlight_timesteps=highlight_timesteps
        )

        return ax

    def print_experiment_summary(self):
        print(
            "\n" + "-" * 10 + "\n",
            "train_length: " + str(self.train_length),
            "test_hilo: " + self.test_hilo,
            "train_hilo: " + self.train_hilo,
            "\ntrain_years:\n",
            [f"{ts.year}-{ts.month}" for ts in self.train_timesteps],
            "\n",
            "test_years:\n",
            [f"{ts.year}-{ts.month}" for ts in self.test_timesteps],
            "\n" + "-" * 10 + "\n",
        )
