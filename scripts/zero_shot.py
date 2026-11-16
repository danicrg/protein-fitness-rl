"""Reproduce ESM2 zero-shot Spearman on ProteinGym assays (thin CLI over pfit.experiment).

    uv run scripts/zero_shot.py                          # starter assays, ESM2 (8M)
    uv run scripts/zero_shot.py --models "ESM2 (8M)" "ESM2 (35M)"
"""

from __future__ import annotations

import argparse

import pandas as pd

from pfit import data
from pfit.experiment import evaluate

RESULTS = data.REPO_ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ESM2 (8M)"])
    ap.add_argument("--assays", nargs="+", default=data.STARTER_ASSAYS)
    args = ap.parse_args()

    table = evaluate(args.models, args.assays, scores_dir=RESULTS / "scores")
    print(table.to_string(index=False))
    out_path = RESULTS / "zero_shot.csv"
    if out_path.exists():  # keep earlier runs for other models/assays
        old = pd.read_csv(out_path)
        key = ["model", "DMS_id"]
        keep = ~old.set_index(key).index.isin(table.set_index(key).index)
        table = pd.concat([old[keep], table], ignore_index=True)
    table.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
