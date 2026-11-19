"""Stage 5: the experiment loop. Score every (model, assay) pair, compare to the leaderboard."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pfit import data
from pfit.esm_score import ESM2Scorer
from pfit.metrics import spearman


def evaluate(models: list[str], dms_ids: list[str], scores_dir=None, device=None) -> pd.DataFrame:
    """One row per (model, assay) with columns:
    model, DMS_id, selection_type, n_mutants, seq_len, spearman_ours, spearman_published,
    delta (ours - published), seconds.

    If scores_dir is given, also write per-mutant scores (mutant, DMS_score, score) to
    scores_dir/<model name with spaces as _>/<DMS_id>.csv.
    Load each model once, not once per assay.
    """
    published = data.published_spearman()
    rows = []
    for model in models:
        scorer = ESM2Scorer(model, device=device)
        for dms_id in dms_ids:
            assay = data.load_assay(dms_id)
            t0 = time.time()
            scores = scorer.score_mutants(assay.target_seq, assay.df["mutant"].tolist())
            out = assay.df[["mutant", "DMS_score"]].assign(score=scores)
            if scores_dir is not None:
                path = Path(scores_dir) / model.replace(" ", "_") / f"{dms_id}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(path, index=False)
            ours = spearman(out["score"], out["DMS_score"])
            ref = float(published.loc[dms_id, model])
            rows.append({
                "model": model,
                "DMS_id": dms_id,
                "selection_type": assay.selection_type,
                "n_mutants": len(out),
                "seq_len": len(assay.target_seq),
                "spearman_ours": round(ours, 3),
                "spearman_published": ref,
                "delta": round(ours - ref, 3),
                "seconds": round(time.time() - t0, 1),
            })
    return pd.DataFrame(rows)
