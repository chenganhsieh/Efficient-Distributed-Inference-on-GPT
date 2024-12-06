# Efficient Distributed Inference on GPT
## Abstract
Efficient distributed inference of large-scale machine learning models presents significant challenges. This project compares three methods to enhance inference efficiency: Knowledge Distillation, Tensor Partitioning, and Model Parallelism. Using the GPT-2 Medium model and the Penn Treebank dataset, we implemented each method under consistent conditions.

## Performance Comparison of Inference Techniques on GPT-2 Medium

| **Method**             | **Batch Size** | **Latency (ms)** | **Throughput (tokens/s)** | **CPU Mem (GB)** | **GPU Mem (GB)** | **Perplexity** | **Accuracy (%)** |
|-------------------------|----------------|------------------:|--------------------------:|-----------------:|-----------------:|---------------:|-----------------:|
| **Baseline**           | 32             | 85.19            | 9576.52                  | 2.90             | 6.68             | 177.94         | 0.72            |
| **Knowledge Distillation** | 32         | 24.48            | 33320.04                | 2.57             | 3.66             | 147.67         | 0.69            |
| **Data Parallelism**    | 64             | 493.57           | 3305.66                 | 4.28             | 14.10            | 177.94         | 0.72            |
| **Pipeline Parallelism** | 64            | 200.47           | 8138.65                 | 3.14             | 8.35             | 177.48         | 0.72            |
| **Tensor Partitioning** | 64            | 214.47           | 7607.46                 | 3.04             | 10.74            | 177.48         | 0.72            |



## Contact:
* Cheng-An Hsieh (chengan2@andrew.cmu.edu)
* Ben Chiang (benchian@andrew.cmu.edu)
* Ling-En Huang (lingenh@andrew.cmu.edu)


