# Review pipeline evaluation (Part 12b)

Relevance labels: labels\qrels_chatgpt.tsv (ChatGPT, declared LLM labels; test set only).

## Reading and claims

| config | set | relevant_per_q | reader_pairs | reader_agree | reader_kappa | reader_bin_agree | reader_bin_kappa | claims | claims_supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| heldout_assign | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| heldout_minicheck | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.846 |
| heldout_single | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| heldout_template | heldout | 6 | 0 | nan | nan | nan | nan | 227 | 0.824 |
| test | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_assign | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_check | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_minicheck | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.874 |
| test_single | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |
| test_template | test | 6.733 | 240 | 0.650 | 0.239 | 0.904 | 0.073 | 677 | 0.821 |

## Synthesis

| config | set | synthesis | verifier | replay_of | questions | points_per_q | sentences_per_q | supported | kept_per_q | copied_kept | claims_used | used_top4 | used_last4 | abstained | thin_evidence | reading_s | synthesis_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | heldout | plan | deberta | - | 12 | 4.583 | 4.583 | 0.927 | 4.250 | 0.181 | 0.652 | 0.710 | 0.575 | 0 | 0 | 76.992 | 25.333 |
| heldout_assign | heldout | plan | deberta | heldout | 12 | 5.417 | 5.417 | 0.908 | 4.917 | 0.257 | 0.807 | 0.813 | 0.800 | 0 | 0 | 76.992 | 23.717 |
| heldout_minicheck | heldout | plan | minicheck | heldout | 12 | 5.500 | 5.500 | 0.924 | 5.083 | 0.229 | 0.844 | 0.833 | 0.857 | 0 | 0 | 76.992 | 48.292 |
| heldout_single | heldout | single | deberta | heldout | 12 | 0 | 5.583 | 0.672 | 3.750 | 0.078 | 0.476 | 0.551 | 0.375 | 0 | 0 | 76.992 | 20.800 |
| heldout_template | heldout | template | deberta | heldout | 12 | 0 | 15.583 | 0.984 | 15.333 | 0.371 | 1.000 | 1.000 | 1.000 | 0 | 0 | 76.992 | 1.208 |
| test | test | plan | deberta | - | 30 | 5.200 | 5.200 | 0.949 | 4.933 | 0.263 | 0.628 | 0.731 | 0.523 | 0 | 0 | 88.267 | 33.050 |
| test_assign | test | plan | deberta | test | 30 | 5.700 | 5.700 | 0.936 | 5.333 | 0.229 | 0.833 | 0.860 | 0.805 | 0 | 0 | 88.267 | 27.433 |
| test_check | test | plan | deberta | test | 30 | 5.167 | 5.167 | 0.948 | 4.900 | 0.290 | 0.629 | 0.713 | 0.545 | 0 | 0 | 88.267 | 26.953 |
| test_minicheck | test | plan | minicheck | test | 30 | 6.267 | 6.267 | 0.915 | 5.733 | 0.247 | 0.824 | 0.865 | 0.781 | 0 | 0 | 88.267 | 60.943 |
| test_single | test | single | deberta | test | 30 | 0 | 5.967 | 0.542 | 3.233 | 0.089 | 0.378 | 0.487 | 0.267 | 0 | 0 | 88.267 | 26.490 |
| test_template | test | template | deberta | test | 30 | 0 | 18.533 | 0.993 | 18.400 | 0.397 | 1.000 | 1.000 | 1.000 | 0 | 0 | 88.267 | 1.180 |
