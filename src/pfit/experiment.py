"""Stage 5: the experiment loop. Score every (model, assay) pair, compare to the leaderboard."""

from __future__ import annotations

import pandas as pd


def evaluate(models: list[str], dms_ids: list[str], scores_dir=None, device=None) -> pd.DataFrame:
    """One row per (model, assay) with columns:
    model, DMS_id, selection_type, n_mutants, seq_len, spearman_ours, spearman_published,
    delta (ours - published), seconds.

    If scores_dir is given, also write per-mutant scores (mutant, DMS_score, score) to
    scores_dir/<model name with spaces as _>/<DMS_id>.csv.
    Load each model once, not once per assay.
    """
    raise NotImplementedError("stage 5")
