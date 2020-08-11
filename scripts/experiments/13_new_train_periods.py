from pathlib import Path
from typing import List, Tuple, Dict
import sys
import itertools


sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
from src.engineer import Engineer


def lstm(ignore_vars, static):
    rnn(
        experiment="one_month_forecast",
        surrounding_pixels=None,
        explain=False,
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=10,
        hidden_size=256,
        # static data
        static=static,
        include_pred_month=True if static is not None else False,
        include_latlons=True if static is not None else False,
        include_prev_y=True if static is not None else False,
    )


def ealstm(ignore_vars, static="features"):
    # -------------
    # EALSTM
    # -------------
    earnn(
        experiment="one_month_forecast",
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        ignore_vars=ignore_vars,
        num_epochs=50,
        early_stopping=10,
        hidden_size=256,
        static_embedding_size=64,
        # static data
        static=static,
        include_pred_month=True,
        include_latlons=True,
        include_prev_y=True,
    )


def calculate_length_of_hi_med_lo_experiment_train_years(
    total_months: int, test_length: int = 12
) -> List[int]:
    # how many months do we have for train/test
    total_train_months = total_months - test_length

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


def sort_by_median_target_var(
    target_data: xr.DataArray,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    target_variable = [v for v in target_data.data_vars][0]

    median_data = target_data.resample(time="M").mean().median(dim=["lat", "lon"])
    median_data = median_data.to_dataframe()

    # sorted low to high
    sorted_df = median_data.sort_values(target_variable)
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
    ):
        self.train_length = train_length
        self.train_hilo = train_hilo
        self.test_hilo = test_hilo
        self.test_length = test_length
        self.sorted_timesteps = sorted_timesteps

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
            sorted_timesteps
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

    def get_experiment_timesteps(
        self, sorted_timesteps: np.array
    ) -> Tuple[np.array, np.array]:
        # 1. split into low, med, high groups
        test_dict = self._calculate_hilo_dict(sorted_timesteps)

        # 2. select randomly the TEST timesteps from these groups
        test_timesteps = np.random.choice(
            test_dict[self.test_hilo], self.test_length, replace=False
        )

        # 3. remove test_timesteps from array to choose train_timesteps
        if self.reshuffle:
            sorted_timesteps_for_training = np.array(
                [ts for ts in self.sorted_timesteps if ts not in test_timesteps]
            )
            train_dict = self._calculate_hilo_dict(sorted_timesteps_for_training)

        else:
            train_dict = test_dict.copy()
            train_dict[self.train_hilo] = np.array(
                [ts for ts in train_dict[self.train_hilo] if ts not in test_timesteps]
            )

        # 4. Choose the train_timesteps
        # get the n training timesteps from the group
        if self.train_length > len(train_dict[self.train_hilo]):
            # if the number of training timesteps is more than the group
            # we need to 'steal' some timesteps from other groups
            train_timesteps = train_dict[self.train_hilo]

            leftover_timesteps = [
                i
                for i in sorted_timesteps
                if (i not in train_timesteps) & (i not in test_timesteps)
            ]

            # iteratively split the array into low, med, high and keep selecting
            # test_timesteps from the correct group until they're all used up
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
                        "not enough {self.train_hilo} timesteps left! Selecting from other groups"
                    )
                    train_timesteps = np.append(
                        train_timesteps, np.random.choice(leftover_timesteps, 1)
                    )
                    leftover_timesteps = [
                        i
                        for i in self.sorted_timesteps
                        if (i not in train_timesteps) & (i not in test_timesteps)
                    ]
                    assert False, "need to select the leftovers in a neater way?"
                    continue
                train_timesteps = np.append(
                    train_timesteps, train_dict[self.train_hilo]
                )
                leftover_timesteps = [
                    i
                    for i in sorted_timesteps
                    if (i not in train_timesteps) & (i not in test_timesteps)
                ]

        else:
            train_timesteps = np.random.choice(
                train_dict[self.train_hilo], self.train_length, replace=self.replacement
            )

        return test_timesteps, train_timesteps

    def plot_experiment_split(self):
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

        fig, ax = plt.subplots()
        title = f"Test: {self.test_hilo} // Train: {self.train_hilo}\nTrainLength:{self.train_length} TestLength:{self.test_length}"
        ax = make_monthly_calendar_plot(df, ax, title=title)

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


if __name__ == "main":
    DEBUG = True

    all_vars = [
        "VCI",
        "precip",
        "t2m",
        "pev",
        "E",
        "SMsurf",
        "SMroot",
        "Eb",
        "sp",
        "tp",
        "ndvi",
        "p84.162",
        "boku_VCI",
        "VCI3M",
        "modis_ndvi",
    ]
    target_vars = ["boku_VCI"]
    dynamic_vars = ["precip", "t2m", "pet", "E", "SMroot", "SMsurf", "VCI3M"]
    static = True

    vars_to_include = target_vars + dynamic_vars
    vars_to_exclude = [v for v in all_vars if v not in vars_to_include]
