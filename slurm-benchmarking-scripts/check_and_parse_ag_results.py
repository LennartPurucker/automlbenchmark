from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Benchmark setup
num_folds = 10
folds = list(range(num_folds))
with open("tabrepo_amlb_tasks.txt") as file:
    tasks = [int(line.strip()) for line in file]
total_combinations = len(tasks) * len(folds)

# Path setup
base_path = Path(__file__).parent.parent
results_csv = pd.read_csv(base_path / "results" / "results.csv")
results_csv = results_csv[~results_csv["result"].isna()]  # Drop failed results
task_to_name_map = results_csv[["task", "id"]].set_index("id").to_dict()["task"]

# -- Check results
for task in tasks:
    task_str = f"openml.org/t/{task}"

    if task_str not in results_csv["id"].unique():
        print(f"Task {task} not found in results.csv")
        continue

    tmp_res = results_csv[results_csv["id"] == task_str]
    has_all_folds = list(np.unique(tmp_res["fold"])) == folds
    if not has_all_folds:
        print(f"Task {task} does not have all folds in results.csv")
        continue

# -- Parse Results
full_results_df = None
all_lb_files = glob.glob(str(base_path / "results" / "**" / "leaderboard.csv"), recursive=True)
# assert len(all_lb_files) == len(results_csv)

task_fold_to_lb_map = {
    (Path(path).parent.parent.parent.parent.name.split("_")[-1].split(".")[0], Path(path).parent.name): path
    for path in all_lb_files
}

# Ensure the index is within the valid range
for index in range(total_combinations):
    # Determine the combination based on the index
    task_index = int(index / num_folds)
    fold_index = index % num_folds

    # Get the corresponding elements from the arrays
    selected_task = tasks[task_index]
    selected_fold = folds[fold_index]

    if (str(selected_task), str(selected_fold)) not in task_fold_to_lb_map:
        print(f"Task {selected_task} fold {selected_fold} not found in leaderboard.csv")
        continue
    lb_path = task_fold_to_lb_map[(str(selected_task), str(selected_fold))]
    df = pd.read_csv(lb_path, index_col=False)
    df["task"] = selected_task
    df["fold"] = selected_fold
    full_results_df = df if full_results_df is None else pd.concat([full_results_df, df], axis=0)

full_results_df.to_csv(base_path / "results" / "full_lb_results.csv", index=False)
