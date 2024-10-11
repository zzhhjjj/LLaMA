"""
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=2    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    tests/test_save_load_weights.py
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    tests/test_save_load_weights.py --tp 4
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun    --nproc_per_node=4    --nnodes=1    --rdzv_backend=c10d    --rdzv_endpoint=localhost:29400    --max_restarts=0    --tee=3    tests/test_save_load_weights.py --tp 2 --dp 2
"""
import argparse
from contextlib import nullcontext
import time
from src.model.llama3 import LLaMA
from src.parallel.initialize import get_data_parallel_group, get_data_parallel_rank, get_data_parallel_world_size, initialize_process_groups, initialize_torch_distributed
from tools.utils import get_num_params, log_info, log_training_steps, num_to_str, setup_logger, load_env_file
import torch 
from dataclasses import dataclass
import os
import torch.nn as nn
import torch.distributed as dist
from dataclasses import asdict
from src.weights.weights import save_weights, load_weights
from datetime import datetime
import shutil


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
    parser.add_argument("--dp", type=int, help="Data parallel size", default=1)
    parser.add_argument("--tp", type=int, help="Tensor parallel size", default=2)
    parser.add_argument("--wandb_log", type=bool, help="Whether to use wandb", default=False)
    return parser.parse_args()

args = get_args()

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

# GPT2
model_config = LLaMAConfig(max_position_embeddings=1024, hidden_dim=768, intermediate_dim=3072, vocab_size=50304, num_key_values=4, num_heads=12, num_layers=12) # https://x.com/karpathy/status/1621578354024677377 still true.  50304 ~30% speedup
parallel_config = parallel_config(model_parallel_size = args.tp, data_parallel_size = args.dp)

initialize_torch_distributed()
initialize_process_groups(model_parallel_size=parallel_config.model_parallel_size, pipeline_parallel_size=1, context_parallel_size=1, data_parallel_size=parallel_config.data_parallel_size)

model = LLaMA(model_config).to(dtype).to(device)
# generate a name for the weights
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
file_prefix = f"./tmp/model_{current_time}/"
save_weights(model, file_prefix)

# Load the weights
new_model = LLaMA(model_config).to(dtype).to(device)
load_weights(new_model, file_prefix)

# Compare the weights
for (name, param), (new_name, new_param) in zip(model.named_parameters(), new_model.named_parameters()):
    assert name == new_name
    assert torch.equal(param, new_param)

torch.distributed.barrier()

# delete 
if dist.get_rank() == 0:
    print(f"Deleting saved weights at {file_prefix}")
    shutil.rmtree(file_prefix)

















