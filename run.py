import argparse
import json
from pathlib import Path
import os

from src import Run, DictWithDefaults


def main(input_args):

    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    default_config_file = base_dir / "pipeline_config/minimal.json"

    with open(default_config_file, "r") as f:
        default_config = json.load(f)

    if input_args.config is not None:
        with open(args.config, "r") as f:
            user_config = json.load(f)
    else:
        user_config = {}

    config = DictWithDefaults(user_config, default_config)

    data_path = Path(config["data"])
    runtask = Run(data_path)
    runtask.run(config, args.run_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the drought prediction pipeline")
    parser.add_argument(
        "--config",
        required=False,
        help="path to configuration file describing the pipeline to be run",
    )
    parser.add_argument(
        "--run-from", default="export", help="Which step to start the pipeline from"
    )
    args = parser.parse_args()
    main(args)
