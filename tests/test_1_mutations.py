"""Stage 1: mutation strings. Run: uv run pytest tests/test_1_mutations.py -v"""

import pytest

from pfit.mutations import Mutation, apply_mutant, parse_mutant

SEQ = "MKTAYIAK"  # positions (1-indexed): M1 K2 T3 A4 Y5 I6 A7 K8


def test_single_mutation_is_zero_indexed():
    assert parse_mutant("K2A", SEQ) == [Mutation("K", 1, "A")]


def test_returns_mutation_namedtuples():
    m = parse_mutant("T3W", SEQ)[0]
    assert (m.wt, m.pos, m.mt) == ("T", 2, "W")


def test_multi_mutation_splits_on_colon():
    assert parse_mutant("M1A:K8R", SEQ) == [Mutation("M", 0, "A"), Mutation("K", 7, "R")]


def test_multi_digit_positions():
    seq = "A" * 120 + "W"
    assert parse_mutant("W121F", seq) == [Mutation("W", 120, "F")]


def test_wrong_wildtype_raises():
    with pytest.raises(ValueError):
        parse_mutant("A2G", SEQ)  # position 2 is K, not A


def test_no_check_without_target():
    assert parse_mutant("A2G") == [Mutation("A", 1, "G")]


def test_offset():
    # offset is the number of the first residue: offset=0 means positions count from 0
    assert parse_mutant("K1A", SEQ, offset=0) == [Mutation("K", 1, "A")]


def test_apply_single_and_multi():
    assert apply_mutant(SEQ, "K2A") == "MATAYIAK"
    assert apply_mutant(SEQ, "M1A:K8R") == "AKTAYIAR"


def test_apply_does_not_change_length():
    assert len(apply_mutant(SEQ, "Y5P")) == len(SEQ)


# --- bounds and error messages --------------------------------------------------
@pytest.mark.parametrize("mutant", ["K0A", "K9A", "K99A"])
def test_out_of_range_position_raises_valueerror(mutant):
    # K0A -> pos -1: Python would silently read the LAST residue
    with pytest.raises(ValueError):
        parse_mutant(mutant, SEQ)


def test_apply_out_of_range_raises_and_never_edits_last_residue():
    with pytest.raises(ValueError):
        apply_mutant(SEQ, "K0A")


def test_empty_target_is_still_checked():
    # "" is falsy, but it was given: the check must run (no position 2 in an empty sequence)
    with pytest.raises(ValueError):
        parse_mutant("Z2A", "")


def test_error_message_names_the_mutant_and_both_letters():
    with pytest.raises(ValueError) as err:
        parse_mutant("M1A:A2G", SEQ)  # position 2 is K, not A
    msg = str(err.value)
    assert "A2G" in msg, "say which mutation failed"
    assert "K" in msg.replace("A2G", ""), "say which letter was actually there"
