"""Stage 2: Spearman from scratch. Run: uv run pytest tests/test_2_metrics.py -v"""

import inspect

import numpy as np
import pytest
from scipy.stats import spearmanr

from pfit import metrics
from pfit.metrics import rankdata, spearman


def test_implementation_does_not_use_scipy():
    assert "scipy" not in inspect.getsource(metrics), "implement it with numpy only"


def test_rankdata_simple():
    assert np.allclose(rankdata([30, 10, 20]), [3, 1, 2])


def test_rankdata_ties_get_average_rank():
    assert np.allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


def test_perfect_and_reversed():
    x = np.arange(10)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_matches_scipy_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=200), rng.normal(size=200)
        assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-9)


def test_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 300), rng.integers(0, 5, 300)  # lots of ties
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-9)


def test_ignores_nans():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([2.0, 1.0, 3.0, np.nan, 5.0])
    assert spearman(a, b) == pytest.approx(spearmanr([1, 2, 5], [2, 1, 5]).statistic)


def test_ignores_nans_without_shifting_ranks():
    # drop the NaN pair BEFORE ranking: what's left is perfectly correlated
    assert spearman([1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
