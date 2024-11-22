# Efficient Distributed Inference on GPT
## Abstract
Efficient distributed inference of large-scale machine learning models presents significant challenges. This project compares three methods to enhance inference efficiency: Knowledge Distillation, Tensor Partitioning, and Model Parallelism. Using the GPT-2 Medium model and the Penn Treebank dataset, we implemented each method under consistent conditions.

## Experiemnt result

| Model              | Average Latency       | Throughput            | Memory Usage (GB) | Perplexity | Token-Level Accuracy |
|--------------------|-----------------------|-----------------------|-------------------|------------|----------------------|
| Baseline           | 222.27 ms/batch       | 921.91 tokens/sec     | 3.15              | 182.12     | 0.72%                |
| Knowledge Distill  | 7.03 ms/query        | 29129.64 queries/sec    | 3.62              | 150.87        | 0.69%                  |
| Model Parallelism  | 21.78 ms/batch        | 9,408.55 tokens/sec   | 2.40              | 182.12     | 0.72%                |
| Tensor Partition   | xx ms/batch           | xx tokens/sec         | xx                | xx         | xx%                  |


## Contact:
* Cheng-An Hsieh(chengan2)
* Ben Chiang
* Ling-En Huang


