"""Stage 2: Spearman rank correlation, numpy only.

ProteinGym's headline metric. Each assay measures fitness in its own units (binding, growth,
stability...), so only the ORDER of the predictions is compared, never their values.
"""

from __future__ import annotations

import numpy as np


def rankdata(x) -> np.ndarray:
    """Ranks starting at 1. Tied values all get the average of the ranks they span.

    rankdata([30, 10, 20]) -> [3, 1, 2];  rankdata([10, 20, 20, 30]) -> [1, 2.5, 2.5, 4]
    """
    x = np.asarray(x, dtype=float)
    argsort = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[argsort[j+1]] == x[argsort[i]]:
            j +=1
        avg = (i + j)/2
        ranks[argsort[i:j+1]] = avg + 1
        i = j + 1
    
    return ranks


def spearman(pred, target) -> float:
    """Pearson correlation of the ranks. Drop pairs where either value is NaN first."""
    pred = np.asarray(pred)
    target = np.asarray(target)

    mask = ~(np.isnan(pred) | np.isnan(target))

    pred_rank = rankdata(pred[mask])
    target_rank = rankdata(target[mask])

    rp = pred_rank - pred_rank.mean()
    rt = target_rank - target_rank.mean()

    return (rp*rt).sum() / np.sqrt((rp**2).sum() * (rt**2).sum())