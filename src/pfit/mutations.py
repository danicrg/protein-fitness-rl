"""Stage 1: parse ProteinGym mutant strings.

ProteinGym writes a mutation as  <wild-type letter><position><mutant letter>,  e.g. ``K2A``:
"position 2 (counting from 1) is K in the natural protein; this variant has A there".
Multiple mutations are joined with colons: ``M1A:K8R``.
"""

from __future__ import annotations

from typing import NamedTuple


class Mutation(NamedTuple):
    wt: str  # wild-type amino acid (one letter)
    pos: int  # 0-indexed position into the target sequence
    mt: str  # mutant amino acid (one letter)


def parse_mutant(mutant: str, target_seq: str | None = None, offset: int = 1) -> list[Mutation]:
    """Split a colon-separated mutant string into Mutations.

    ``offset`` is the number of the first residue in the string's convention (ProteinGym: 1),
    so ``pos = written_position - offset``.
    If ``target_seq`` is given, raise ValueError when its letter at ``pos`` isn't ``wt``.
    """
    raise NotImplementedError("stage 1")


def apply_mutant(target_seq: str, mutant: str, offset: int = 1) -> str:
    """Return target_seq with every mutation in ``mutant`` applied."""
    raise NotImplementedError("stage 1")
