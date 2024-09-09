# Througput Benchmark
## TP compare with FSDP on LLaMA2 7b  
1. First time benchmark for TP=4   
train_config = train_config(lr=3e-4,accumulation_steps=1, batch_size=4)    
model_config = LLaMAConfig(max_position_embeddings=4096, intermediate_dim=11008, vocab_size=32000, num_key_values=32) # LLaMA 2 7b   
Tokens/s/GPU: 7.3K compared with 7.5K      
FLOPs = 6 * num_params * num_tokens = 6 * 6.74B * 7.3K = 295 * 10e12    
MFU = FLOPs/tp_size/hardward_flops = 295 * 10e12 / 4 / 989 * 10e12 # only 6% ?? I made a mistake somewhere. Current MFU: 0.36   
Ref: https://pytorch.org/blog/maximizing-training/   



