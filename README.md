# Protein Fitness RL

Aligning protein language models to experimental fitness with preference optimization and RL, evaluated without leakage.

## Plan

| Rung | Question | Status |
|---|---|---|
| 1 | Can I reproduce ESM2 zero-shot fitness prediction on [ProteinGym](https://proteingym.org)? | in progress: building the scoring pipeline stage by stage |
| 2 | How much do random train/test splits overstate supervised fitness models? (sequence-identity-aware splits) | |
| 3 | Does DPO/GRPO against a learned fitness oracle improve generated sequences, and when does it reward-hack? | |
| 4 | Workshop paper and a ProteinGym leaderboard submission | |
| 5 | Wet-lab validation through an open protein design competition | |

## Quickstart

```bash
uv sync
uv run pytest -q                                              # staged tests: tests/test_<n>_*.py
uv run scripts/zero_shot.py                                   # 9 starter assays, ESM2 (8M)
uv run scripts/zero_shot.py --models "ESM2 (8M)" "ESM2 (35M)"
```

Data (ProteinGym v1.3, about 43 MB zipped) downloads on first use into `data/` (git-ignored; override with `PFIT_DATA`). Results land in `results/zero_shot.csv`, side by side with the published leaderboard Spearman for the same model and assay.

## Method notes

- Zero-shot score: ESM2 masked marginals, the strategy ProteinGym uses for ESM models. For each mutated position, mask it in the wild type and take `log p(mutant) - log p(wild type)`, summed over the mutations.
- Sequences longer than 1022 residues are scored in a window centred on each position. That's a simplification of ProteinGym's optimal-window logic.
- Starter assays: 9 small, single-mutant assays covering the five ProteinGym selection types (Binding, Activity, Expression, Stability, Organismal fitness). See `STARTER_ASSAYS` in [src/pfit/data.py](src/pfit/data.py).

## Layout

- `src/pfit/`: library code (data, mutations, ESM scoring, metrics)
- `scripts/`: one script per experiment
- `results/`: summary tables, committed (per-mutant scores are git-ignored)
- `posts/`: write-ups, one per rung

## References

- Notin et al. 2023, ProteinGym: Large-scale benchmarks for protein fitness prediction and design
- Meier et al. 2021, Language models enable zero-shot prediction of the effects of mutations on protein function
- Lin et al. 2023, Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM2)
- Widatalla, Rafailov, Hie 2024, Aligning protein generative models with experimental fitness via direct preference optimization (ProteinDPO)
