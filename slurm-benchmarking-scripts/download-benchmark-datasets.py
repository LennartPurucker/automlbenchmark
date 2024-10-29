from __future__ import annotations

import openml
from pathlib import Path
import argparse
import tqdm

REGRESSION_AMLB_SUITE = 269
CLASSIFICATION_AMLB_SUITE = 271


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("../openml-cache"),
        help="Directory to save the datasets",
    )
    parser.add_argument("--action", choices=["download", "list"])
    args = parser.parse_args()

    openml.config.set_cache_directory(str(args.directory.resolve().absolute()))

    classification = openml.study.get_suite(CLASSIFICATION_AMLB_SUITE)
    assert classification.tasks is not None

    regression = openml.study.get_suite(REGRESSION_AMLB_SUITE)
    assert regression.tasks is not None

    match args.action:
        case "download":
            for task in tqdm.tqdm(classification.tasks + regression.tasks):
                openml.tasks.get_task(task, download_data=True, download_qualities=True)
        case "list":
            for task in classification.tasks + regression.tasks:
                print(task)
