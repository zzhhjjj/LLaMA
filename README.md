# LLaMA

Welcome to **LLaMA**, my library for training and fine-tuning the LLaMA model. I find it helpful to implement something from scratch to gain a better understanding. I hope the simplicity of this repo could potentially serve as a good starting point for beginners.

## Features

Currently, this library supports:

1. Flash Attention, Triton RMSNorm, Flash RoPE (Triton/CUDA acceleration)
2. KV Cache
3. Tensor Parallelism
4. DDP with bucket 
## Experience
1. Speedup/Loss benchmark results under LLaMA/tools/benchmark

## Coming Soon

I'm actively working on integrating the following features:

1. Training on real data
2. More benchmarks    
3. Zero Optimizer    
