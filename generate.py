import os
from src.model.llama3 import LLaMA
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from src.weights.weights import load_weights_to_dict, copy_weights_to_model
from transformers import AutoTokenizer

# to match the output of transformers model, set MERGED_QKV_WEIGHT to 0 is necessary
os.environ['DATA_TYPE'] = 'bfloat16' # bfloat16/float32
os.environ['MERGED_QKV_WEIGHT'] = '0' # 1/0
os.environ['MERGED_GATE_UP_WEIGHT'] = '0' # 1/0
os.environ['ATTENTION'] = 'FLASH' # SDPA/FLASH

# set device and dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

@dataclass
class LLaMAConfig:
    batch_size: int = 1
    max_position_embeddings: int = 8192
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    vocab_size = 128256
    num_key_values: int = 8
    num_heads: int = 32
    num_layers: int = 32
    rope_theta: float = 500000.0
    torch_dtype: str = 'bfloat16'
    rms_norm_eps: float = 1e-5

## initialize model 
config = LLaMAConfig()
model = LLaMA(config)

## load weights 
weights_directory_path = ".cache/models/Meta-Llama-3-8B"
all_tensors = load_weights_to_dict(weights_directory_path)
copy_weights_to_model(model, all_tensors)
model.to(dtype).to(device)

## Tokenizer
pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

## generation
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
max_new_tokens = 300
inputs['input_ids']= model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, use_cache=True)

print("Input prompt:", input_text)
print("Generated text:", tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)[len(input_text):])  # remove the input text from the generated text





