"""Stage 2: Spearman rank correlation, numpy only (no scipy in this file).

ProteinGym's headline metric. Each assay measures fitness in its own units (binding, growth,
stability...), so only the ORDER of the predictions is compared, never their values.
"""

from __future__ import annotations

import numpy as np


def rankdata(x) -> np.ndarray:
    """Ranks starting at 1. Tied values all get the average of the ranks they span.

    rankdata([30, 10, 20]) -> [3, 1, 2];  rankdata([10, 20, 20, 30]) -> [1, 2.5, 2.5, 4]
    """
    raise NotImplementedError("stage 2")


def spearman(pred, target) -> float:
    """Pearson correlation of the ranks. Drop pairs where either value is NaN first."""
    raise NotImplementedError("stage 2")
