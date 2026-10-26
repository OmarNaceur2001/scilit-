# Review pipeline evaluation (Part 12b)

Relevance labels: labels\qrels_chatgpt.tsv (ChatGPT, declared LLM labels; test set only).

## Reading and claims

| config | set | relevant_per_q | reader_pairs | reader_agree | reader_kappa | reader_bin_agree | reader_bin_kappa | claims | claims_supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| heldout_single | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| heldout_template | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| test | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_check | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_single | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_template | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |

## Synthesis

| config | set | synthesis | verifier | replay_of | questions | points_per_q | sentences_per_q | supported | kept_per_q | copied_kept | abstained | thin_evidence | reading_s | synthesis_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | heldout | plan | deberta | - | 12 | 4.583 | 4.583 | 0.927 | 4.250 | 0.181 | 0 | 0 | 76.992 | 25.333 |
| heldout_single | heldout | single | deberta | heldout | 12 | 0 | 5.583 | 0.672 | 3.750 | 0.078 | 0 | 0 | 76.992 | 20.800 |
| heldout_template | heldout | template | deberta | heldout | 12 | 0 | 15.583 | 0.984 | 15.333 | 0.371 | 0 | 0 | 76.992 | 1.208 |
| test | test | plan | deberta | - | 30 | 5.200 | 5.200 | 0.949 | 4.933 | 0.263 | 0 | 0 | 88.267 | 33.050 |
| test_check | test | plan | deberta | test | 30 | 5.167 | 5.167 | 0.948 | 4.900 | 0.290 | 0 | 0 | 88.267 | 26.953 |
| test_single | test | single | deberta | test | 30 | 0 | 5.967 | 0.542 | 3.233 | 0.089 | 0 | 0 | 88.267 | 26.490 |
| test_template | test | template | deberta | test | 30 | 0 | 18.533 | 0.993 | 18.400 | 0.397 | 0 | 0 | 88.267 | 1.180 |
