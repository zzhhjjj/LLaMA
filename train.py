"""Training script for LLaMA model.
Command to run this script:
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    train.py
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run -- --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 --max_restarts=0 train.py
Why CUDA_DEVICE_MAX_CONNECTIONS=1: 
1. https://github.com/NVIDIA/Megatron-LM/blob/8b0a9b32470e84eb0dbefc557b4e4d4ca342cc65/megatron/core/tensor_parallel/layers.py#L445-L446,
2. https://zhuanlan.zhihu.com/p/706805407



"""
import argparse
from contextlib import nullcontext
import time
from src.model.llama3 import LLaMA
from src.parallel.initialize import get_data_parallel_group, get_data_parallel_rank, get_data_parallel_world_size, initialize_process_groups, initialize_torch_distributed
from tools.utils import get_num_params, log_info, log_training_steps, num_to_str, setup_logger, load_env_file
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
from torch.utils.data import DistributedSampler
from src.parallel.data_parallel.data_parallel import DataParallel
from src.parallel.data_parallel.data_parallel_bucket import DataParallel as DataParallel_Bucket
from dataclasses import asdict

# Set env variables
load_env_file('.env')

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

# Parse the command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, help="Data parallel size", default=4)
    parser.add_argument("--tp", type=int, help="Tensor parallel size", default=1)
    parser.add_argument("--wandb_log", type=bool, help="Whether to use wandb", default=False)
    return parser.parse_args()

args = get_args()

# scaler not support for bfloat16 yet. 
# use_amp = True 
# ctx = nullcontext() if device.type == 'cpu' else torch.autocast(device_type=device.type , dtype=dtype, enabled=use_amp)
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Training Hyerparameters
@dataclass
class train_config:
    num_epochs: int = 5
    total_steps: int = 300  # -1 for full dataset
    accumulation_steps: int = 4
    lr: float = 3e-4
    batch_size: int = 64
    adam_weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-10

@dataclass
class parallel_config:
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 1

## Initialize model, loss function, and optimizer
## LLaMA3 8b model
## Note: For the convergence speed is slow compared to 180M model. Need more test.
# # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json
@dataclass
class LLaMAConfig:
    max_position_embeddings: int = 8192
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    vocab_size: int = 131072 
    num_key_values: int = 8
    num_heads: int = 32
    num_layers: int = 32
    rope_theta: float = 500000.0
    torch_dtype: str = 'bfloat16'
    rms_norm_eps: float = 1e-5
# train_config = train_config(lr=3e-4,accumulation_steps=1, batch_size=4)
# LLaMA2 7b 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model_config = LLaMAConfig(max_position_embeddings=4096, intermediate_dim=11008, vocab_size=32000, num_key_values=32) 
# LLaMA3 8b
# model_config = LLaMAConfig() 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# GPT2
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model_config = LLaMAConfig(max_position_embeddings=1024, hidden_dim=768, intermediate_dim=3072, vocab_size=50304, num_key_values=4, num_heads=12, num_layers=12) # https://x.com/karpathy/status/1621578354024677377 still true.  50304 ~30% speedup
train_config = train_config(lr=3e-4, accumulation_steps=4, batch_size=64, total_steps=300)
parallel_config = parallel_config(model_parallel_size = args.tp, data_parallel_size = args.dp)

dataset = load_dataset("roneneldan/TinyStories", split='train')
tokenized_dataset = tokenize_dataset(dataset, tokenizer, text_column_name = 'text', sequence_length = model_config.max_position_embeddings, dataset_processing_num_proc_per_process = 64)

# Initialize torch distributed, process groups
initialize_torch_distributed()
initialize_process_groups(model_parallel_size=parallel_config.model_parallel_size, pipeline_parallel_size=1, context_parallel_size=1, data_parallel_size=parallel_config.data_parallel_size)
world_size = dist.get_world_size()
master_process = dist.get_rank() == 0 # master process is the process do the logs

# DataLoader
shuffle = False
if parallel_config.data_parallel_size > 1:
    sampler = DistributedSampler(tokenized_dataset, num_replicas=get_data_parallel_world_size(), rank=get_data_parallel_rank(), shuffle=shuffle) # without distributed sampler, the data will be duplicated in each GPU of the data parallel group.  
else:
    sampler = None
dataloader = get_dataloader(tokenized_dataset, train_config.batch_size, shuffle=shuffle, sampler=sampler)

# Model
model = LLaMA(model_config).to(dtype).to(device)

# DDP
if get_data_parallel_world_size() > 1:
    # model = DataParallel(model, process_group=get_data_parallel_group()) # easiest implementation
    # model = DataParallel_Bucket(model, process_group=get_data_parallel_group(), bucket_cap_mb=25) # DDP implementation with bucket
    model = torch.nn.parallel.DistributedDataParallel(model, process_group=get_data_parallel_group())

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # reduction='mean' by default 
optimizer = optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.adam_weight_decay, betas=(train_config.adam_beta1, train_config.adam_beta2), eps=train_config.adam_eps)

# Initialize the logger 
log_path = "tools/benchmark/loss/loss.txt"
logger = setup_logger(log_path)

# Log some information 
log_info(model_config,train_config,parallel_config, logger)
log_training_steps(train_config.num_epochs, train_config.total_steps, dataloader)
num_params = get_num_params(model,logger)
total_tokens_processed = 0
tokens_per_step = train_config.batch_size * model_config.max_position_embeddings * train_config.accumulation_steps * parallel_config.data_parallel_size

# wandb to save the training logs
if args.wandb_log and master_process:
    import wandb
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # run_name = f"GBS_{num_to_str(tokens_per_step)}_dp{parallel_config.data_parallel_size}_tp{parallel_config.model_parallel_size}_cp{parallel_config.context_parallel_size}_pp{parallel_config.pipeline_parallel_size}_{current_time}"
    run_name = f"GBS_{num_to_str(tokens_per_step)}_dp{parallel_config.data_parallel_size}_tp{parallel_config.model_parallel_size}_{current_time}"
    wandb.init(
        project="llama",
        name=run_name,
        config={
            "seed": seed,
            **asdict(train_config), 
            **asdict(parallel_config),  
            **asdict(model_config), 
        }
    )

# Initilize the step counter
step = 0
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
            tokens_processed = B * T # B is batch size, T is sequence length
            total_tokens_processed += tokens_processed
            
            # Print the loss for each step
            if step % 10 == 0 and master_process:
                print(f'Step [{step}], Loss: {loss.item():.4f}, Tokens/s : {total_tokens_processed / (time.time() - start_time):.2f}')
                if step <= 50:
                    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            # profiler
            if step >= 50:
                break       
else:
    gradient_accumulation_counter = 0
    for epoch in range(train_config.num_epochs):
        for data in dataloader:
            input_ids, label_ids = data['input_ids'].to(device) , data['label_ids'].to(device)
            B, T = input_ids.size()
            label_ids = label_ids.view(B*T)
            
            # disable the gradient sync for all but the last micro_step.
            # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/train.py#L293-L298
            model.require_backward_grad_sync = (gradient_accumulation_counter == train_config.accumulation_steps - 1)
            
            # Forward pass with mixed precision
            outputs = model(input_ids)
            outputs = outputs.view(B*T, -1) # adjust shape for the loss
            loss = criterion(outputs, label_ids) / train_config.accumulation_steps 

            # Backward pass and optimize
            loss.backward()
            
            gradient_accumulation_counter += 1
            
            # Gradient accumulation: only step and zero gradients every 'accumulation_steps'
            if gradient_accumulation_counter % train_config.accumulation_steps == 0:
                optimizer.step()         # Perform optimizer step
                optimizer.zero_grad()    # Zero the parameter gradients
                
                # Need to set the bucket to zero. I can avoid this by adding if condition in DDP class, but it's necessary for PP anyway. 
                # https://github.com/pytorch/pytorch/blob/c0deec120fc1c97cf6439772df5d5bebfa736e4c/torch/nn/parallel/distributed.py#L1096-L1104
                if hasattr(model, 'reset'):
                    model.reset()
            
                # Increment step counter
                step += 1
                gradient_accumulation_counter = 0
            
                total_tokens_processed += tokens_per_step
            
                # Print the loss for each step
                log_interval = 1
                if step % log_interval == 0:
                    # Avg loss from all DP processes
                    if parallel_config.data_parallel_size > 1:
                        loss_tensor = torch.tensor([loss.item()], device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    avg_loss = loss_tensor.item() if parallel_config.data_parallel_size > 1 else loss.item()
                    
                    current_time = time.time()
                    time_taken = current_time - start_time
                    tokens_per_second = tokens_per_step * log_interval / time_taken
                    flops_per_gpu = model.get_flops(train_config.accumulation_steps, time_taken, num_params)/world_size 
                    start_time = current_time
                    
                    if master_process:                            
                        logger.info(f"Step [{step}], Epoch [{epoch+1}/{train_config.num_epochs}], Loss: {avg_loss * train_config.accumulation_steps:.4f}, "
                            f"Global batch size: {num_to_str(tokens_per_step)}, "
                            f"Accumulated tokens: {num_to_str(total_tokens_processed)}, "
                            f"Tokens/s: {num_to_str(tokens_per_second)}, "
                            f"Tokens/s/GPU: {num_to_str(tokens_per_second/world_size)}, "
                            f"FLOPS/GPU: {num_to_str(flops_per_gpu)}, " 
                            f"MFU: {num_to_str(flops_per_gpu/989e10)}% " # H100 GPU bfloat16 peak flops： 989e12
                        )
                    
                        memory_reserved_gb = torch.cuda.memory_reserved(device) / 1024 ** 3
                        logger.info(f"Memory Reserved: {memory_reserved_gb:.2f} GB")

                        # Log the training metrics to wandb as well 
                        if args.wandb_log:
                            log_data = {
                                "step": step,
                                "epoch": epoch + 1,
                                "loss": avg_loss * train_config.accumulation_steps,
                                "global_batch_size": tokens_per_step,
                                "total_tokens_processed": total_tokens_processed,
                                "tokens_per_second": tokens_per_second,
                                "tokens_per_second_per_gpu": tokens_per_second / world_size,
                                "flops_per_gpu": flops_per_gpu,
                                "mfu": flops_per_gpu / 989e10,  # H100 GPU bfloat16 peak flops： 989e12
                                "memory_reserved_gb": memory_reserved_gb,
                            }
                            # log_data["memory_reserved_gb"] = memory_reserved_gb
                            wandb.log(log_data) 

                # Check if the training should be stopped
                if train_config.total_steps != -1 and step >= train_config.total_steps:
                    finished = True
                    break
        if finished:
            break
        logger.info(f'Epoch [{epoch+1}/{train_config.num_epochs}], Loss: {loss.item():.4f}')
    if master_process:
        logger.info("Training complete.")





