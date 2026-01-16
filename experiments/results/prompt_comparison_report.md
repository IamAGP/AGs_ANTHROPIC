# Prompt Evolution: Side-by-Side Comparison

## Summary Metrics

| Prompt Variant | Accuracy | Avg Precision | Avg Recall | Answer w/o Retrieval | Tool Usage Rate |
|---------------|----------|---------------|------------|---------------------|----------------|
| misleading | 93.3% | 0.06666666666666667 | 0.2 | 11/15 | 26.7% |
| broken | 100.0% | 0.33333333333333326 | 1.0 | 0/15 | 100.0% |
| default | 100.0% | 0.33333333333333326 | 1.0 | 0/15 | 100.0% |
| verbose | 100.0% | 0.33333333333333326 | 1.0 | 0/15 | 100.0% |

## Failure Mode Breakdown

| Prompt Variant | Success | Retrieval Failure | Prompt Following | Answer w/o Retrieval | No Search |
|---------------|---------|-------------------|------------------|---------------------|----------|
| misleading | 3 | 0 | 0 | 11 | 1 |
| broken | 15 | 0 | 0 | 0 | 0 |
| default | 15 | 0 | 0 | 0 | 0 |
| verbose | 15 | 0 | 0 | 0 | 0 |
