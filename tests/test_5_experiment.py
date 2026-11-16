"""Stage 5: the experiment loop. Run: uv run pytest tests/test_5_experiment.py -v"""

import torch

from pfit.experiment import evaluate

ASSAYS = ["B2L11_HUMAN_Dutta_2010_binding-Mcl-1", "OTU7A_HUMAN_Tsuboyama_2023_2L2D"]


def test_evaluate_table(tmp_path):
    t = evaluate(["ESM2 (8M)"], ASSAYS, scores_dir=tmp_path, device=torch.device("cpu"))
    assert len(t) == 2
    for col in ["model", "DMS_id", "selection_type", "n_mutants", "spearman_ours",
                "spearman_published", "delta"]:
        assert col in t.columns
    assert (t["delta"].abs() < 0.002).all()
    assert len(list(tmp_path.rglob("*.csv"))) == 2  # per-mutant scores written
