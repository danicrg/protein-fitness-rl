"""Stage 3: ProteinGym data. Downloads about 45 MB on first run.
Run: uv run pytest tests/test_3_data.py -v"""

import pytest

from pfit import data
from pfit.mutations import apply_mutant, parse_mutant

B2L11 = "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"


def test_reference_has_all_assays():
    ref = data.reference()
    assert len(ref) == 217
    for col in ["DMS_id", "DMS_filename", "target_seq", "coarse_selection_type"]:
        assert col in ref.columns


def test_published_spearman_indexed_by_assay():
    pub = data.published_spearman()
    assert pub.index.name == "DMS_id"
    assert pub.loc[B2L11, "ESM2 (8M)"] == pytest.approx(0.174)


def test_load_assay_fields():
    a = data.load_assay(B2L11)
    assert a.dms_id == B2L11
    assert a.selection_type == "Binding"
    assert len(a.df) == 170
    assert {"mutant", "mutated_sequence", "DMS_score"} <= set(a.df.columns)


def test_every_mutant_matches_the_wild_type():
    # ties stage 1 to real data: each mutant's wild-type letter is really in the target
    for dms_id in ["BLAT_ECOLX_Jacquier_2013", "OTU7A_HUMAN_Tsuboyama_2023_2L2D"]:
        a = data.load_assay(dms_id)
        for mutant, mutated in zip(a.df["mutant"], a.df["mutated_sequence"]):
            parse_mutant(mutant, a.target_seq)
            assert apply_mutant(a.target_seq, mutant) == mutated


def test_unknown_assay_raises_keyerror():
    with pytest.raises(KeyError):
        data.load_assay("NOT_A_REAL_ASSAY")
