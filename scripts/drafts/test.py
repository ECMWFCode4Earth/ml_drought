import pickle
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")
    all_errors = pickle.load((data_dir / "RUNOFF/all_errors.pkl").open("rb"))
    all_metrics = pickle.load((data_dir / "RUNOFF/all_metrics.pkl").open("rb"))

    # TEST THAT INVERSE FUNCITON WORKING
    test_model = [k for k in all_errors.keys()][0]
    test_metric = [k for k in all_metrics.keys()][0]
    assert all(all_errors[test_model][test_metric].dropna() == all_metrics[test_metric][test_model].dropna())

    ##
    import xarray as xr
    import sys
    sys.path.append("/home/tommy/ml_drought")
    from scripts.drafts.calculate_error_scores import calculate_seasonal_errors, get_metric_dataframes_from_output_dict
    from collections import defaultdict

    save = True
    all_preds = xr.open_dataset()
    seasonal_errors = calculate_seasonal_errors(all_preds)
    #  calculate seasonal metrics
    seasonal_metrics = defaultdict(dict)
    for season in ["DJF", "MAM", "JJA", "SON"]:
        seasonal_metrics[season] = get_metric_dataframes_from_output_dict(
            seasonal_errors[season]
        )

    if save:
        import pickle

        pickle.dump(
            seasonal_errors, (data_dir / "RUNOFF/seasonal_errors.pkl").open("wb")
        )
        pickle.dump(
            seasonal_metrics, (data_dir / "RUNOFF/seasonal_metrics.pkl").open("wb")
        )
