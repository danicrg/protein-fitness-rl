# Reproducing ESM2 zero-shot fitness prediction on ProteinGym (draft)

Target: rung 1, about Dec 13.

## Outline

1. **The question.** Can a protein language model trained only on natural sequences predict which mutations break a protein? What is a deep mutational scan (DMS), and what does Spearman against it mean?
2. **Setup.** ProteinGym v1.3, 9 starter assays across the 5 selection types, ESM2 8M and 35M, masked-marginal scoring.
3. **Result.** Our Spearman against the published Spearman per assay (table from `results/zero_shot.csv`). Where they match, and where they don't and why.
4. **What surprised me.** Bigger isn't always better (in the published numbers 650M beats 8M on some assays and loses on others). Activity against abundance on the same protein (VKOR1). Stability assays against organismal fitness.
5. **What I'd do next.** The supervised splits (rung 2).

## Notes while running

- 2026-11-16: the first run matches the published Spearman on all 18 (model, assay) pairs to 3 decimals (ESM2 8M mean 0.201, 35M mean 0.382). Scoring is under 3 s per assay on an M-series Mac.
- VKOR1 is the same protein measured two ways: activity (8M: 0.016, 35M: 0.298) and abundance (8M: 0.251, 35M: 0.443). Why is activity so much harder? Look up what each assay measures.
- 35M beats 8M on 8 of 9 assays; B2L11 binding goes the other way (0.174 to 0.123), but it has only 170 mutants. How noisy is Spearman at n=170? Bootstrap a CI.
