"""Training script for LLaMA model.
Command to run this script:
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    train.py
Why CUDA_DEVICE_MAX_CONNECTIONS=1: 
1. https://github.com/NVIDIA/Megatron-LM/blob/8b0a9b32470e84eb0dbefc557b4e4d4ca342cc65/megatron/core/tensor_parallel/layers.py#L445-L446,
2. https://zhuanlan.zhihu.com/p/706805407



"""
import sys
import time
import logging
from src.model.llama3 import LLaMA
from src.parallel.tensor_parallel.initialize import initialize_process_groups, initialize_torch_distributed
from tools.debug.logs import RedirectOutput, log_model_info, log_training_steps, num_to_str, setup_logger
import torch 
from dataclasses import dataclass
from src.weights.weights import load_weights_to_dict, copy_weights_to_model
from transformers import AutoTokenizer
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datasets import load_dataset
from src.data.data import tokenize_dataset, get_dataloader


# to match the output of transformers model, set MERGED_QKV_WEIGHT to 0 is necessary
os.environ['DATA_TYPE'] = 'bfloat16' # bfloat16/float32
os.environ['MERGED_QKV_WEIGHT'] = '0' # 1/0
os.environ['MERGED_GATE_UP_WEIGHT'] = '0' # 1/0
os.environ['TRITONRMSNORM'] = '1'
os.environ['FLASH_ROPE'] = '1'
os.environ['ATTENTION'] = 'FLASH' # SDPA/FLASH
os.environ['USE_PROFILER'] = '0' # 1/0
profiler_output_dir = ".cache/profile"

# set device and dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

## Initialize model, loss function, and optimizer
## GPT2 base model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
sequence_length = 1024
@dataclass
class LLaMAConfig:
    batch_size: int = 1
    max_position_embeddings: int = sequence_length
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    vocab_size = 50304 #  https://x.com/karpathy/status/1621578354024677377 still true.  50304 ~30% speedup
    num_key_values: int = 4
    num_heads: int = 12
    num_layers: int = 12
    rope_theta: float = 500000.0
    torch_dtype: str = 'bfloat16'
    rms_norm_eps: float = 1e-5
batch_size = 64

## LLaMA3 8b model
## Note: without TP, I cannot train real 8b model! Priority is to support TP!
# # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# sequence_length = 8192
# @dataclass
# class LLaMAConfig:
#     # batch_size: int = 1
#     max_position_embeddings: int = sequence_length
#     hidden_dim: int = 4096
#     intermediate_dim: int = 14336
#     vocab_size = 131072 
#     num_key_values: int = 8
#     num_heads: int = 32
#     num_layers: int = 32
#     rope_theta: float = 500000.0
#     torch_dtype: str = 'bfloat16'
#     rms_norm_eps: float = 1e-5
# batch_size = 2

dataset = load_dataset("roneneldan/TinyStories", split='train')
tokenized_dataset = tokenize_dataset(dataset, tokenizer, text_column_name = 'text', sequence_length = sequence_length, dataset_processing_num_proc_per_process = 64)

# DataLoader
shuffle = False
dataloader = get_dataloader(tokenized_dataset, batch_size, shuffle)

# Initialize torch distributed, process groups
initialize_torch_distributed()
initialize_process_groups(model_parallel_size=1, pipeline_parallel_size=1, context_parallel_size=1, data_parallel_size=1)
world_size = dist.get_world_size()

# Model
config = LLaMAConfig()
model = LLaMA(config).to(dtype).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # or any other suitable loss function
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Training loop
step = 0
num_epochs = 5
total_steps = 300 # -1 for full dataset
accumulation_steps = 4
tokens_per_step = batch_size * sequence_length * accumulation_steps

# Initialize the logger 
log_path = "tools/benchmark/loss/loss.txt"
logger = setup_logger(log_path)

log_model_info(model,logger)
log_training_steps(num_epochs, total_steps, dataloader)

total_tokens_processed = 0
start_time = time.time()  # Start time measurement

use_profiler = os.getenv('USE_PROFILER', '0') == '1'
if use_profiler:
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=True,
        profile_memory=True,
        with_flops = True,
        with_modules=True,
        with_stack=True
    ) 
else:
    profiler = None

if use_profiler:
    with profiler:
        print("Profile with torch profiler")

        for data in dataloader:
            profiler.step()
            
            input_ids, label_ids = data['input_ids'].to(device) , data['label_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # adjust shape for the loss
            B, T, C = outputs.size()
            outputs = outputs.view(B*T, C)
            label_ids = label_ids.view(B*T)
            
            # Compute the loss
            loss = criterion(outputs, label_ids)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Increment step counter
            step += 1
            
            # Print the loss for each step
            tokens_processed = B * T  # B is batch size, T is sequence length
            total_tokens_processed += tokens_processed
            
            # Print the loss for each step
            if step % 10 == 0 and dist.get_rank() == 0:
                print(f'Step [{step}], Loss: {loss.item():.4f}, Tokens/s : {total_tokens_processed / (time.time() - start_time):.2f}')
                if step <= 50:
                    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            # profiler
            if step >= 50:
                break       
else:
    gradient_accumulation_counter = 0
    for epoch in range(num_epochs):
        for data in dataloader:
            input_ids, label_ids = data['input_ids'].to(device) , data['label_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # adjust shape for the loss
            B, T, C = outputs.size()
            outputs = outputs.view(B*T, C)
            label_ids = label_ids.view(B*T)
            
            # Compute the loss
            loss = criterion(outputs, label_ids) / accumulation_steps

            # Backward pass and optimize
            loss.backward()
            
            gradient_accumulation_counter += 1
            
            # Gradient accumulation: only step and zero gradients every 'accumulation_steps'
            if gradient_accumulation_counter % accumulation_steps == 0:
                optimizer.step()         # Perform optimizer step
                optimizer.zero_grad()    # Zero the parameter gradients
            
                # Increment step counter
                step += 1
                gradient_accumulation_counter = 0
            
                total_tokens_processed += tokens_per_step
            
                # Print the loss for each step
                log_interval = 1
                if step % log_interval == 0 and dist.get_rank() == 0:
                    current_time = time.time()
                    tokens_per_second = tokens_per_step * log_interval / (current_time - start_time)
                    start_time = current_time
                    
                    logger.info(f"Step [{step}], Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item() * accumulation_steps:.4f}, "
                        f"Global batch size: {num_to_str(tokens_per_step)}, "
                        f"Accumulated tokens: {num_to_str(total_tokens_processed)}, "
                        f"Tokens/s: {num_to_str(tokens_per_second)}, "
                        f"Tokens/s per GPU: {num_to_str(tokens_per_second/world_size)} "
                    )
                    if step <= 10:
                        logger.info(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
                    if total_steps != -1 and step >= total_steps:
                        finished = True
                        break
        if finished:
            break
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    logger.info("Training complete.")





