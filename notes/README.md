# What I have learned
1. Increasing the vocabulary size from 50,257 to 50,304 can result in a 30% speedup on the GPT model.
2. Despite having significantly fewer FLOPs, RMSNorm and Rotary embedding are as slow as MLP.
3. Be cautious with the precision of RoPE.
4. Merging matrix multiplication appears to be problematic due to significant accumulated errors.   
5. When training small model (190M params), moving data from CPU to GPU took a large amount of time: 56%. batch_size, seq_length = (64, 1024).    