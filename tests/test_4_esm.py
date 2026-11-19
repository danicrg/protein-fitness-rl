"""Stage 4: ESM2 masked-marginal scoring. Downloads ESM2 8M (about 30 MB) on first run.
Run one sub-stage at a time, e.g.: uv run pytest tests/test_4_esm.py -v -k 4a"""

import random

import pytest
import torch

from pfit import data
from pfit.esm_score import MAX_TOKENS, ESM2Scorer
from pfit.metrics import spearman

AA = "ACDEFGHIKLMNPQRSTVWY"
SEQ = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQT"
    "LGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPF"
    "LPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSAVGWAATVGL"
)


@pytest.fixture(scope="module")
def scorer():
    return ESM2Scorer("ESM2 (8M)", device=torch.device("cpu"))


def random_mutants(seq, n, k=1, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pos = sorted(rng.sample(range(len(seq)), k))
        out.append(":".join(f"{seq[p]}{p + 1}{rng.choice(AA.replace(seq[p], ''))}" for p in pos))
    return out


# 4a: masked log-probs ---------------------------------------------------------
def test_4a_logprobs_are_a_distribution(scorer):
    lp = scorer.masked_logprobs(SEQ, [0, 10, len(SEQ) - 1])
    assert set(lp) == {0, 10, len(SEQ) - 1}
    for row in lp.values():
        assert row.shape == (len(scorer.tok),)
        assert torch.logsumexp(row, 0).item() == pytest.approx(0.0, abs=1e-4)


def test_4a_masking_changes_the_prediction(scorer):
    # with position 10 masked, the model can't just copy the letter that is there
    lp = scorer.masked_logprobs(SEQ, [10])[10]
    ids = scorer.tok(SEQ, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        unmasked = torch.log_softmax(scorer.model(input_ids=ids).logits[0, 11], -1)
    wt = scorer.tok.convert_tokens_to_ids(SEQ[10])
    assert unmasked[wt] > lp[wt], "unmasked should be more confident in the wild type"


def test_4a_batch_matches_one_at_a_time(scorer):
    # each copy in the batch must mask only its own position
    batch = scorer.masked_logprobs(SEQ, [0, 10, 50])
    for p in (0, 10, 50):
        alone = scorer.masked_logprobs(SEQ, [p])[p]
        assert torch.allclose(batch[p], alone, atol=1e-4), f"position {p} differs in the batch"


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="no accelerator"
)
def test_4a_runs_on_the_default_device():
    gpu = ESM2Scorer("ESM2 (8M)")  # default device: cuda or mps
    lp = gpu.masked_logprobs(SEQ, [10])[10]
    assert lp.shape == (len(gpu.tok),)


# 4b: naive scoring, one forward pass per mutated position ---------------------
def test_4b_naive_single_is_logratio(scorer):
    lp = scorer.masked_logprobs(SEQ, [4])[4]
    aa = scorer.tok.convert_tokens_to_ids
    expected = (lp[aa("W")] - lp[aa(SEQ[4])]).item()
    assert scorer.score_mutant_naive(SEQ, f"{SEQ[4]}5W") == pytest.approx(expected, abs=1e-4)


def test_4b_naive_multi_is_sum_of_singles(scorer):
    a, b = f"{SEQ[2]}3G", f"{SEQ[30]}31P"
    both = scorer.score_mutant_naive(SEQ, f"{a}:{b}")
    singles = scorer.score_mutant_naive(SEQ, a) + scorer.score_mutant_naive(SEQ, b)
    assert both == pytest.approx(singles, abs=1e-4)


def test_4b_wrong_wild_type_raises(scorer):
    # a mutant whose wild-type letter doesn't match the sequence is a data error, not a score
    wrong = "W" if SEQ[0] != "W" else "A"
    with pytest.raises(ValueError):
        scorer.score_mutant_naive(SEQ, f"{wrong}1A")


def test_4b_returns_a_python_float(scorer):
    assert type(scorer.score_mutant_naive(SEQ, f"{SEQ[4]}5W")) is float


# 4c: batched scoring, one forward pass per distinct position ------------------
def test_4c_batched_equals_naive(scorer):
    muts = random_mutants(SEQ, 25) + random_mutants(SEQ, 10, k=2, seed=1)
    fast = scorer.score_mutants(SEQ, muts)
    slow = [scorer.score_mutant_naive(SEQ, m) for m in muts]
    assert fast == pytest.approx(slow, abs=1e-3)


def test_4c_returns_python_floats(scorer):
    out = scorer.score_mutants(SEQ, random_mutants(SEQ, 5))
    assert all(type(x) is float for x in out), "return float(...), not tensors"


def test_4c_wrong_wild_type_raises(scorer):
    wrong = "W" if SEQ[0] != "W" else "A"
    with pytest.raises(ValueError):
        scorer.score_mutants(SEQ, [f"{SEQ[4]}5W", f"{wrong}1A"])


def test_4c_one_masked_row_per_distinct_position_in_bounded_batches(scorer):
    # memory grows with batch x length^2: one unchunked batch on a 1000-residue protein is ~80 GB
    muts = random_mutants(SEQ, 150) + random_mutants(SEQ, 30, k=2, seed=1)
    distinct = {int(t[1:-1]) for m in muts for t in m.split(":")}
    batch_sizes = []

    def record(module, args, kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        batch_sizes.append(ids.shape[0])

    handle = scorer.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        scorer.score_mutants(SEQ, muts)
    finally:
        handle.remove()
    assert sum(batch_sizes) == len(distinct), "one masked copy per distinct position, no repeats"
    assert max(batch_sizes) <= 64, f"one forward pass saw {max(batch_sizes)} sequences: chunk it"


# 4d: windows for sequences longer than the model's context --------------------
@pytest.mark.parametrize("seq_len", [50, 1022, 1023, 1500, 3000])
def test_4d_window_properties(scorer, seq_len):
    width = MAX_TOKENS - 2
    for pos in [0, 1, seq_len // 2, seq_len - 2, seq_len - 1]:
        s, e = scorer._window(pos, seq_len)
        assert 0 <= s <= pos < e <= seq_len
        assert e - s == min(width, seq_len)


def test_4d_long_sequence_batched_equals_naive(scorer):
    long_seq = (SEQ * 8)[:1500]
    muts = random_mutants(long_seq, 6, seed=2) + [f"{long_seq[0]}1A", f"{long_seq[-1]}1500A"]
    muts = [m if m[0] != m[-1] else m[:-1] + "W" for m in muts]
    fast = scorer.score_mutants(long_seq, muts)
    slow = [scorer.score_mutant_naive(long_seq, m) for m in muts]
    assert fast == pytest.approx(slow, abs=1e-3)


# 4e: ground truth, ProteinGym's own published numbers -------------------------
@pytest.mark.parametrize(
    "dms_id", ["B2L11_HUMAN_Dutta_2010_binding-Mcl-1", "ENVZ_ECOLI_Ghose_2023"]
)
def test_4e_reproduces_leaderboard(scorer, dms_id):
    a = data.load_assay(dms_id)
    ours = spearman(scorer.score_mutants(a.target_seq, a.df["mutant"].tolist()), a.df["DMS_score"])
    published = data.published_spearman().loc[dms_id, "ESM2 (8M)"]
    assert ours == pytest.approx(published, abs=0.0015)
