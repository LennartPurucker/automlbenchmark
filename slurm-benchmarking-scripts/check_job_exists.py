from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

task = sys.argv[1]
fold_id = sys.argv[2]
framework = sys.argv[3]

path = Path(__file__).parent.parent / "results" / "results.csv"

if not path.exists():
    print("False")
    sys.exit(0)

res = pd.read_csv(path)

result_score = res[
    (res["id"] == f"openml.org/t/{task}") & (res["fold"] == int(fold_id)) & (res["framework"] == framework)
]["result"]

if result_score.empty:
    print("False")
    sys.exit(0)

print(str(any(~result_score.isna())))
