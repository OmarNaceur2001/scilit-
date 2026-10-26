# Bronze evaluation (classical components only)

## Table 1 — Results per component on the frozen test set

| Component | Lot | Bronze method | Trivial baseline | Metric | Baseline | Bronze | N test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Retrieval | D | BM25 | random ranking | Recall@10 (↑) | 0.009 | 0.652 | 30 queries |
| Retrieval | D | BM25 | random ranking | MRR (↑) | 0.03 | 0.95 | 30 queries |
| Sections | B | heading lexicon + inheritance | majority (appendix) | macro-F1 (↑) | 0.025 | 0.779 | 75 headings |
| Claims (sentence roles) | B | cue patterns | majority (method) | macro-F1 (↑) | 0.095 | 0.357 | 1135 sentences |
| Claims (sentence roles) | C | TF-IDF 1-2 grams + logistic regression | majority (method) | macro-F1 (↑) | 0.095 | 0.691 | 1135 sentences |
| Review by template | D | BM25 + quoted sentences | — | references that exist (↑) | — | 1.0 | 240 references |
| Out-of-scope rule | E | IDF-weighted coverage ≥ 0.557 (chosen on dev) | always answer | balanced accuracy (↑) | 0.5 | 0.9 | 30 in + 5 out |

Rule components on dev and test separately (macro-F1):

| Component | dev | test |
| --- | --- | --- |
| Sections (rules) | 0.789 (n=75) | 0.779 (n=75) |
| Claims (cue patterns) | 0.278 (n=2003) | 0.357 (n=1135) |

Bronze pipeline on the 30 test queries: 7.93 quoted sentences per review, 0.583 of the papers read judged relevant, latency 0.42 s (max 0.99 s); in-scope questions answered 30/30, out-of-scope questions abstained 4/5.

## Table 2 — Results per language (same 30 queries)

| Component | Metric | EN | FR |
| --- | --- | --- | --- |
| Retrieval BM25 | Recall@10 | 0.652 | 0.047 |
| Retrieval BM25 | MRR | 0.95 | 0.095 |

## Table 3 — Error analysis (all errors in reports/bronze_errors.csv)

| Component | Category | Count |
| --- | --- | --- |
| claims (patterns) | no cue: defaulted to background (true: method) | 222 |
| claims (patterns) | no cue: defaulted to background (true: result) | 101 |
| claims (patterns) | objective cue fired on a method sentence | 62 |
| claims (patterns) | no cue: defaulted to background (true: objective) | 44 |
| claims (patterns) | no cue: defaulted to background (true: other) | 40 |
| claims (patterns) | result cue fired on a method sentence | 28 |
| claims (patterns) | result cue fired on a background sentence | 22 |
| claims (patterns) | method cue fired on a background sentence | 18 |
| claims (patterns) | objective cue fired on a background sentence | 14 |
| claims (patterns) | objective cue fired on a result sentence | 12 |
| claims (patterns) | method cue fired on a result sentence | 8 |
| claims (patterns) | method cue fired on an objective sentence | 4 |
| claims (patterns) | result cue fired on an objective sentence | 4 |
| retrieval | relevant paper ranked below 10 | 98 |
| retrieval | query words matched, different topic | 25 |
| retrieval | French query: no shared word with English papers | 25 |
| retrieval | match inside a related-work section | 4 |
| sections | heading word unknown to the lexicon -> inherited parent type | 13 |
| sections | heading word unknown to the lexicon -> other | 8 |
| sections | lexicon keyword points to another type (method) | 8 |
| sections | lexicon keyword points to another type (results) | 4 |
| sections | lexicon keyword points to another type (related_work) | 3 |
| sections | inside appendix: forced to appendix | 2 |
| sections | lexicon keyword points to another type (experiments) | 2 |
| sections | lexicon keyword points to another type (discussion) | 1 |

## Frozen test files

| File | SHA-256 (16) | First commit |
| --- | --- | --- |
| labels/queries.tsv | a4d1161dbec1afe0 | 2026-10-24 |
| labels/qrels_chatgpt.tsv | d17555bace7c8926 | 2026-10-24 |
| labels/heldout.tsv | e9fb0d9ceecdc35c | 2026-10-26 |
| configs/splits.json | c8044603f5cee806 | 2026-10-22 |
| labels/dev_scope.tsv | 2ec3a09a38d8d381 | not committed |
