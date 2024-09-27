from contextlib import contextmanager
import os
import sys
from functools import wraps
import time
import logging
import torch
from src.parallel.tensor_parallel.initialize import get_model_parallel_group, get_model_parallel_world_size, get_pipeline_parallel_group, get_pipeline_parallel_world_size
import torch.distributed as dist
from dataclasses import asdict

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

def log_info(model_config, train_config, parallel_congig, logger):
    log_config(model_config, logger, "LLaMAConfig:")
    log_config(train_config, logger,"TrainConfig:")
    log_config(parallel_congig, logger, "ParallelConfig:")

def get_num_params(model,logger):
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_params_tensor = torch.tensor(total_params, dtype=torch.float32).to(device)
    dist.all_reduce(total_params_tensor, group=get_model_parallel_group(), async_op=False, op=dist.ReduceOp.SUM)  # TP
    dist.all_reduce(total_params_tensor, group=get_pipeline_parallel_group(), async_op=False, op=dist.ReduceOp.SUM)  # PP 
    total_params = total_params_tensor.item()

    if dist.get_rank() == 0:
        # Print the number of parameters
        logger.info(f"Total number of parameters: {num_to_str(total_params)}\n")
    return total_params

def log_config(config, logger, message):
    if dist.get_rank() == 0:
        logger.info(message)
        config_dict = asdict(config)  # Convert the dataclass to a dictionary
        for key, value in config_dict.items():
            logger.info(f"  {key}: {value}")
        logger.info("")  # Add an empty line after logging

def log_training_steps(num_epochs, total_steps, dataloader):
    if dist.get_rank() == 0:
        if total_steps != -1:
            assert num_epochs * len(dataloader) >= total_steps, "The number of steps is greater than the number of steps in the dataloader"
            print(f"Training for {total_steps} steps")
        else:
            print(f"Training for {num_epochs} epochs")
            
def num_to_str(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"
    
def load_env_file(path='.env'):
    """
    Loads environment variables from a file and sets them in the OS environment.
    """
    try:
        with open(path) as f:
            for line in f:
                # Ignore comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                # Split the line by the first '=' symbol
                key, value = line.split('=', 1)
                # Set the environment variable
                os.environ[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
