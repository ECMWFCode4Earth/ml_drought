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
