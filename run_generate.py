from src.llama3 import LLaMA
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Config:
    batch_size: int = 1
    seq_length: int = 12
    hidden_dim: int = 128
    intermediate_dim: int = 512
    vocab_size = 1024
    num_queries: int = 16
    num_key_values: int = 4
    num_heads: int = 16
    num_layers: int = 16

config = Config()

model = LLaMA(config).to(device)

input_id = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length)).to(device)
print(input_id)

output_logits = model(input_id, attention_mask=None)
print(output_logits.shape)







