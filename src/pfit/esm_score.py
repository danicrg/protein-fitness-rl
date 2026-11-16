"""Stage 4: zero-shot fitness scoring with ESM2 masked marginals (Meier et al. 2021).

Score of a mutant = sum over its mutated positions i of
    log p(x_i = mutant letter | rest) - log p(x_i = wild-type letter | rest)
where position i is replaced by <mask> in the WILD-TYPE sequence.
Negative score: "evolution would rather keep the original letter here", so likely damaging.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

# ProteinGym leaderboard column -> Hugging Face checkpoint (given)
ESM2_CHECKPOINTS = {
    "ESM2 (8M)": "facebook/esm2_t6_8M_UR50D",
    "ESM2 (35M)": "facebook/esm2_t12_35M_UR50D",
    "ESM2 (150M)": "facebook/esm2_t30_150M_UR50D",
    "ESM2 (650M)": "facebook/esm2_t33_650M_UR50D",
    "ESM2 (3B)": "facebook/esm2_t36_3B_UR50D",
}
MAX_TOKENS = 1024  # ESM2 context, including the <cls> and <eos> tokens the tokenizer adds


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ESM2Scorer:
    def __init__(self, name: str = "ESM2 (8M)", device: torch.device | None = None):
        # given: load tokenizer + model. Explore them before stage 4a (see LEARN.md).
        ckpt = ESM2_CHECKPOINTS.get(name, name)
        self.name = name
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = EsmForMaskedLM.from_pretrained(ckpt).eval().to(self.device)

    # 4d
    def _window(self, pos: int, seq_len: int) -> tuple[int, int]:
        """Residue window [start, end) the model sees when scoring position ``pos``.

        Whole sequence if it fits in MAX_TOKENS - 2 residues; otherwise exactly MAX_TOKENS - 2
        residues, centred on ``pos`` as far as the sequence ends allow.
        """
        raise NotImplementedError("stage 4d")

    # 4a
    @torch.no_grad()
    def masked_logprobs(self, seq: str, positions: list[int]) -> dict[int, torch.Tensor]:
        """{position: log-probs over the whole vocabulary (len(self.tok)) with that position
        masked}. One masked copy of the sequence per position; batch them."""
        raise NotImplementedError("stage 4a")

    # 4b
    @torch.no_grad()
    def score_mutant_naive(self, target_seq: str, mutant: str) -> float:
        """Masked-marginal score of one mutant: one forward pass per mutated position.
        Slow but obviously correct; stage 4c is checked against it."""
        raise NotImplementedError("stage 4b")

    # 4c
    def score_mutants(self, target_seq: str, mutants: list[str]) -> list[float]:
        """Same scores as score_mutant_naive, for many mutants, with ONE masked forward pass per
        distinct mutated position (an assay has ~19 mutants per position)."""
        raise NotImplementedError("stage 4c")
