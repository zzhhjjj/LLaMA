from contextlib import contextmanager
import os
import sys
from functools import wraps
import time

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

def log_model_num_params(model):
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Print the number of parameters
    if total_params >= 1e9:
        print(f"Total number of parameters: {total_params / 1e9:.2f} Billion")
    else:
        print(f"Total number of parameters: {total_params / 1e6:.2f} Million")
    return total_params