from contextlib import contextmanager
import os
import sys
from functools import wraps
import time
import logging
from src.parallel.tensor_parallel.initialize import get_model_parallel_group, get_model_parallel_world_size, get_pipeline_parallel_group, get_pipeline_parallel_world_size
import torch.distributed as dist

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{name} took {elapsed_time:.2f} seconds to execute")

class RedirectOutput:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        if self.file_path:
            self.f = open(self.file_path, 'w')
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = self.f
            sys.stderr = self.f

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_path:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            self.f.close()

def print_env_variables():
    print("MERGED_QKV_WEIGHT= ", os.getenv('MERGED_QKV_WEIGHT', '1'))
    print("MERGED_GATE_UP_WEIGHT= ", os.getenv('MERGED_GATE_UP_WEIGHT', '1'))
    print("DATA_TYPE= ", os.getenv('DATA_TYPE', 'bfloat16'))
    print("Attention= ", os.getenv('ATTENTION', 'SDPA'))

def get_log_file_path():
    path = '/fsx/haojun/LLaMA/.cache/logs'
    file_name = ''
    file_name += 'merged_qkv' if os.getenv('MERGED_QKV_WEIGHT', '1') == '1' else 'seperate_qkv'
    file_name += '_merged_gate_up' if os.getenv('MERGED_GATE_UP_WEIGHT', '1') == '1' else '_seperate_gate_up'
    file_name += '_sdpa' if os.getenv('ATTENTION', 'SDPA') == 'SDPA' else '_flash'
    file_name += '_bf16' if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else '_fp32'
    return os.path.join(path, file_name + '.txt')

def setup_logger(file_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create a file handler if a file path is provided
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Optional: add a console handler if you want to still log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def log_model_info(model, logger):
    log_config(model.model_config, logger)
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    dist.all_reduce(total_params, group=get_model_parallel_group(), async_op=False, op=dist.ReduceOp.SUM)  # TP
    dist.all_reduce(total_params, group=get_pipeline_parallel_group(), async_op=False, op=dist.ReduceOp.SUM)  # PP 

    if dist.get_rank() == 0:
        # Print the number of parameters
        if total_params >= 1e9:
            logger.info(f"Total number of parameters: {total_params / 1e9:.2f} Billion\n")
        else:
            logger.info(f"Total number of parameters: {total_params / 1e6:.2f} Million\n")
    return total_params


def log_config(config, logger):
    if dist.get_rank() == 0:
        logger.info(f"LLaMAConfig:")
        # logger.info(f"  batch_size: {config.batch_size}")
        logger.info(f"  max_position_embeddings: {config.max_position_embeddings}")
        logger.info(f"  hidden_dim: {config.hidden_dim}")
        logger.info(f"  intermediate_dim: {config.intermediate_dim}")
        logger.info(f"  vocab_size: {config.vocab_size}")
        logger.info(f"  num_key_values: {config.num_key_values}")
        logger.info(f"  num_heads: {config.num_heads}")
        logger.info(f"  num_layers: {config.num_layers}")
        logger.info(f"  rope_theta: {config.rope_theta}")
        logger.info(f"  torch_dtype: {config.torch_dtype}")
        logger.info(f"  rms_norm_eps: {config.rms_norm_eps}\n")

def log_training_steps(num_epochs, total_steps, dataloader):
    if dist.get_rank() == 0:
        if total_steps != -1:
            assert num_epochs * len(dataloader) >= total_steps, "The number of steps is greater than the number of steps in the dataloader"
            print(f"Training for {total_steps} steps")
        else:
            print(f"Training for {num_epochs} epochs")
            
def num_to_str(num, precision=2):
    if num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return str(num)