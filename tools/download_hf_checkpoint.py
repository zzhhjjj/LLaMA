import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    local_dir = "../.cache/models/Meta-Llama-3-8B",
    local_dir_use_symlinks=False,
    ignore_patterns=["original/*"]
)

# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3-8B"
# cache_dir = "/fsx/haojun/.HF_CACHE/llama3"

# os.makedirs(cache_dir, exist_ok=True)

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the model
# model = AutoModelForCausalLM.from_pretrained(model_name)

# model.save_pretrained(f"{cache_dir}/model")
# tokenizer.save_pretrained(f"{cache_dir}/tokenizer")
