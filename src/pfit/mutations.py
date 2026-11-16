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
    mutations = []
    for mutation in mutant.split(":"):
        digit_end = len(mutation) - 1
        wt = mutation[0]
        pos = int(mutation[1:digit_end]) - offset
        mt = mutation[-1]

        if target_seq and target_seq[pos] != wt:
            raise ValueError("target write does not match")
        
        mutations.append(
            Mutation(wt, pos, mt)
        )
    return mutations


def apply_mutant(target_seq: str, mutant: str, offset: int = 1) -> str:
    """Return target_seq with every mutation in ``mutant`` applied."""
    new_seq = list(target_seq)

    mutations = parse_mutant(mutant, target_seq=target_seq, offset=offset)
    for mutation in mutations: 
        new_seq[mutation.pos] = mutation.mt

    return ''.join(new_seq)
