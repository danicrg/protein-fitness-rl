"""Stage 4: zero-shot fitness scoring with ESM2 masked marginals (Meier et al. 2021).

Score of a mutant = sum over its mutated positions i of
    log p(x_i = mutant letter | rest) - log p(x_i = wild-type letter | rest)
where position i is replaced by <mask> in the WILD-TYPE sequence.
Negative score: "evolution would rather keep the original letter here", so likely damaging.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from pfit.mutations import parse_mutant

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
        if seq_len < MAX_TOKENS - 2:
            return 0, seq_len

        start = pos - (MAX_TOKENS-2)//2
        end = pos + (MAX_TOKENS-2) //2
        if start < 0:
            end -= start
            start -= start
        if end > seq_len:
            start -= end - seq_len
            end = seq_len
        
        return max(0, start), end

    # 4a
    @torch.no_grad()
    def masked_logprobs(self, seq: str, positions: list[int]) -> dict[int, torch.Tensor]:
        """{position: log-probs over the whole vocabulary (len(self.tok)) with that position
        masked}. One masked copy of the sequence per position; batch them."""
        tokenized_seq = self.tok(seq, return_tensors='pt')['input_ids'].to(self.device)
        ids = tokenized_seq.repeat(len(positions), 1)
        for i, position in enumerate(positions):
            ids[i][position + 1] = self.tok.mask_token_id
        # Chunking: at most batch_size copies per forward pass, so memory stays bounded
        batch_size = 32
        rows = []
        for start in range(0, len(positions), batch_size):
            partial_logits = self.model(ids[start:start+batch_size]).logits  # (chunk, L+2, vocab)
            for j, p in enumerate(positions[start:start+batch_size]):
                rows.append(partial_logits[j, p + 1])  # copy j of this chunk, at its mask
        logits = torch.stack(rows)  # (n, vocab), row i is positions[i]

        masked_logprobs = {}
        for i, position in enumerate(positions):
            masked_logprobs[position] = torch.log_softmax(logits[i], dim=-1)
        return masked_logprobs
            

    # 4b
    @torch.no_grad()
    def score_mutant_naive(self, target_seq: str, mutant: str) -> float:
        """Masked-marginal score of one mutant: one forward pass per mutated position.
        Slow but obviously correct; stage 4c is checked against it."""
        total_score = 0
        for submutant in mutant.split(":"):
            position = int(submutant[1:len(submutant)-1]) -1
            if target_seq[position] != submutant[0]:
                raise ValueError(f"The mutation {submutant} is incompatible with the sequence")
            ids = self.tok.convert_tokens_to_ids
            original_id = ids(submutant[0])
            destination_id = ids(submutant[-1])
            masked_logprobs = self.masked_logprobs(target_seq, [position])[position]
            # we sum because in log space sum equals multiplication in linear. 
            total_score += masked_logprobs[destination_id] - masked_logprobs[original_id]
        return total_score.item()



    # 4c
    def score_mutants(self, target_seq: str, mutants: list[str]) -> list[float]:
        """Same scores as score_mutant_naive, for many mutants, with ONE masked forward pass per
        distinct mutated position (an assay has ~19 mutants per position)."""
        positions = set()
        parsed_mutants = list()
        ids = self.tok.convert_tokens_to_ids
        for mutant in mutants: 
            mutations = parse_mutant(mutant, target_seq=target_seq)
            parsed_mutants.append(mutations)
            for mutation in mutations:
                positions.add(mutation.pos)
        positions = list(positions)
        masked_logprobs = self.masked_logprobs(target_seq, positions)
        scores = []
        for mutations in parsed_mutants:
            score = 0
            for submutant in mutations:
                lps = masked_logprobs[submutant.pos]
                score += lps[ids(submutant.mt)] - lps[ids(submutant.wt)]
            scores.append(score.item())
        return scores