# SciLit — Experiment log

Every number shown in the presentation is in this file, with the command that produced it.
Test sets are frozen (hashes in `reports/bronze_eval.md`, section "Frozen test files").
Labels made by an LLM are declared as such.

---

## exp_001 — Initial setup
- **Date**: 2026-10-08
- **Component**: infrastructure
- **Result**: project structure in place

## exp_002 — Corpus and parsing
- **Date**: 2026-10-22
- **Component**: corpus (Lot A)
- **Config**: OpenAlex keyword search + 41 seed papers + citation snowball, 13 themes; arXiv metadata check; PDF quality gate; PyMuPDF rule parser
- **Result**: 1,549 candidates → 1,174 PDFs parsed (0 errors, 65 without a paper structure) → 1,109 structured papers; 331,566 sentences (282 per paper); 68,510 passages indexed
- **Command**: `corpus.collect`, `corpus.normalize`, `corpus.fetch_pdfs`, `corpus.validate`, `corpus.parse_pdf` → `reports/parse_report.md`

## exp_003 — Frozen test set for retrieval
- **Date**: 2026-10-24
- **Component**: data (Lot A)
- **Config**: 30 test questions (`labels/queries.tsv`, English + French versions); relevance labels 0/1/2 by ChatGPT on pooled results (`labels/qrels_chatgpt.tsv`, LLM labels)
- **Agreement**: 52 pairs labelled by a human (`labels/qrels.tsv`) vs ChatGPT: kappa 0.42 on 3 levels, 0.64 on relevant / not relevant (87% agreement)

## exp_004 — Retrieval methods
- **Date**: 2026-10-24
- **Component**: retrieval (Lot D)
- **Config**: 30 test questions, paper-level ranking, `evaluation.retrieval_eval`
- **Result (test)**:

| System | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| BM25 (EN) | 0.652 | 0.950 | 0.823 |
| Dense BGE-M3 (EN) | 0.621 | 0.888 | 0.786 |
| Hybrid RRF (EN) | 0.654 | 0.883 | 0.818 |
| Hybrid + rerank (EN) | 0.684 | 0.928 | 0.827 |
| Hybrid + rerank (FR) | 0.486 | 0.864 | 0.634 |
| BM25, fixed-size chunks | 0.513 | 0.893 | 0.673 |

- **Significance**: hybrid + rerank vs BM25 on nDCG@10, paired bootstrap p = 0.42 (not significant)

## exp_005 — Sentence roles, official CSAbstruct split (superseded by exp_007)
- **Date**: 2026-10-24
- **Component**: claims / representations (Lots B, C)
- **Result (test)**: rules 0.344, TF-IDF 0.724, SciBERT 0.760 ± 0.005 macro-F1 (3 seeds)

## exp_006 — Reproduced published result: citation function (SciCite)
- **Date**: 2026-10-24
- **Component**: reproduction
- **Config**: SciBERT fine-tuned, 3 seeds (13, 42, 2024), model chosen on dev; `extraction.citfunc`
- **Result (test)**: TF-IDF 0.799; SciBERT **0.855 ± 0.017** macro-F1; published 0.8549 (Beltagy et al., 2019); paired bootstrap vs TF-IDF p = 0.000
- **Gap**: +0.0001, within one standard deviation across seeds

## exp_007 — Data leakage found in CSAbstruct, classifiers re-run on clean splits
- **Date**: 2026-10-25
- **Component**: data (Lot A), claims (Lots B, C)
- **Finding**: 172 of the 1,349 official test sentences (12.7%) also appear in train; TF-IDF scores 94% on them
- **Fix**: `roles.dedupe` removes from dev and test every sentence already seen (23 dev, 214 test removed); SciCite: 14 dev, 0 test removed
- **Result (clean test, 1,135 sentences)**: majority class 0.095, cue rules 0.357, TF-IDF 0.691, SciBERT **0.749 ± 0.005** (micro 0.746), bootstrap SciBERT vs TF-IDF p ≤ 0.001; dev 0.59–0.61 (dev labels have lower annotator confidence: 0.63 vs 0.83 on test)
- **Citation function (clean)**: 0.855 ± 0.017, unchanged

## exp_008 — Review, first design (single LLM call + repair)
- **Date**: 2026-10-25
- **Component**: generation (beyond Bronze)
- **Result (30 test questions)**: claims supported 81.6%; synthesis sentences supported 59.6% before repair, 61.7% after; repairs accepted 4 of 74
- **Diagnosis**: wrong claim cited, contrasts not stated by any source, filler forced by the output schema

## exp_009 — Review, plan-then-write vs single call, same reading cards
- **Date**: 2026-10-26
- **Component**: generation (beyond Bronze)
- **Config**: `review_eval run` once per question set, then `replay` on the saved cards; held-out set = 12 questions written after the design was fixed (`labels/heldout.tsv`)
- **Result**:

| Set | Plan-then-write supported | Single call supported | p | Kept / review (plan vs single) | p |
| --- | --- | --- | --- | --- | --- |
| Test (30) | 0.952 | 0.529 | < 0.0001 | 4.93 vs 3.23 | < 0.0001 |
| Held-out (12) | 0.929 | 0.675 | 0.0002 | 4.25 vs 3.75 | 0.16 |

- **Run-to-run noise** (same cards, replayed): 0.949 vs 0.948 supported
- **Note**: test questions were inspected during design; the held-out result is the clean one

## exp_010 — Planner: assign every claim
- **Date**: 2026-10-26
- **Component**: generation (beyond Bronze)
- **Finding**: the previous planner used 88% of the claims of the first paper listed and 38% of the last ones, for papers equally relevant
- **Result (test / held-out)**: claims used 0.833 / 0.807 (was 0.628 / 0.652); top-4 vs last-4 papers 0.860 vs 0.805 (was 0.731 vs 0.523); supported 0.936 / 0.908 (was 0.952 / 0.929, p = 0.26 / 0.30, not significant)

## exp_011 — Verifier comparison
- **Date**: 2026-10-26
- **Component**: verification (beyond Bronze)
- **Config**: 113 (sentence, sources) items: 53 real pairs from the runs + 60 hard cases (numbers changed or swapped, entity swapped, finding reversed, unsupported clause, qualifier removed, same topic but unstated); labels by ChatGPT (LLM labels); threshold chosen on the calibration half, results on the test half (56 items)
- **Result (test half)**:

| Verifier | False acceptance | False rejection | Balanced accuracy | Precision (supported) |
| --- | --- | --- | --- | --- |
| DeBERTa-v3 NLI, threshold 0.5 | 0.333 | 0.438 | 0.615 | 0.692 |
| DeBERTa-v3 NLI, calibrated 0.2 | 0.375 | 0.375 | 0.625 | 0.690 |
| Bespoke-MiniCheck-7B | **0.083** | **0.156** | **0.880** | **0.931** |

- **Known issue**: 20 hard cases (unsupported clause, qualifier removed) were badly generated and do not test their type
- **Decision**: MiniCheck becomes the default verifier

## exp_012 — Review with MiniCheck
- **Date**: 2026-10-26
- **Component**: generation + verification (beyond Bronze)
- **Result (test / held-out)**: claims accepted 0.874 / 0.846; synthesis sentences accepted 0.915 / 0.924; kept per review 5.73 / 5.08; claims used 0.824 / 0.844; synthesis time 61 s / 48 s per question

## exp_013 — Bronze evaluation (classical components only)
- **Date**: 2026-10-26
- **Component**: all Bronze components (Lots A–D)
- **Command**: `python -m scilit.evaluation.bronze_eval run` → `reports/bronze_eval.md`, `reports/bronze_errors.csv`
- **Result**: BRONZE_RESULTS
