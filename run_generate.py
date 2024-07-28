from src.model.llama3 import LLaMA
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    

config = LLaMAConfig()

model = LLaMA(config).to(device)

input_id = torch.randint(0, config.vocab_size, (config.batch_size, 30)).to(device)
print(input_id[0])
print("input_id: ",input_id.shape)

output_logits = model(input_id, attention_mask=None)
print("output_logits: ",output_logits.shape)







