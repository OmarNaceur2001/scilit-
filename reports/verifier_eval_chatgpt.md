# Verifier evaluation (Part 11): ChatGPT (LLM labels) labels

113 items (real pairs from the runs + hard cases with a known way to be wrong), labelled by ChatGPT; gold = agreed labels (113 items). Reported on the test half (56 items).

| verifier | threshold | n | false acceptance | false rejection | balanced acc. | precision (supported) |
| --- | --- | --- | --- | --- | --- | --- |
| deberta | 0.2 | 56 | 0.375 | 0.375 | 0.625 | 0.69 |
| deberta | 0.5 | 56 | 0.333 | 0.438 | 0.615 | 0.692 |
| minicheck | - | 56 | 0.083 | 0.156 | 0.88 | 0.931 |

Hard cases the labellers judged NOT supported, accepted anyway (all items, threshold 0.5):

| type | deberta | minicheck |
| --- | --- | --- |
| entity | 1/6 | 0/6 |
| extra | 1/1 | 1/1 |
| number_changed | 0/6 | 0/6 |
| number_swapped | 1/3 | 0/3 |
| qualifier | 1/1 | 0/1 |
| reversed | 1/9 | 1/9 |
| topic | 2/10 | 0/10 |
