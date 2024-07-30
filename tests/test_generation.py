"""
This script compare the output logits/generated token of my LLaMA model with the transformers LLaMMA model.
"""

from src.model.llama3 import LLaMA
import torch 
from dataclasses import dataclass
from src.weights.weights import load_weights_to_dict, copy_weights_to_model
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
weights_directory_path = "/fsx/haojun/LLaMA/.cache/models/Meta-Llama-3-8B"
all_tensors = load_weights_to_dict(weights_directory_path)
copy_weights_to_model(model, all_tensors)
model.to(device)

## Tokenizer
pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

## Reference model
transformer_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
# transformer_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)

## generation
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
max_new_tokens = 50

for i in range(max_new_tokens):
    output_logits = transformer_model(**inputs)['logits']
    my_output_logits = model(**inputs, debug_arguments=None)
    torch.testing.assert_close(my_output_logits,output_logits,rtol=1e-4,atol=1e-4) # check if the output logits are close
    next_token_id = torch.argmax(output_logits[:, -1, :], dim=-1)
    my_next_token_id = torch.argmax(my_output_logits[:, -1, :], dim=-1)
    assert next_token_id == my_next_token_id, 'detect different prediction' # check if the prediction is the same
    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id.unsqueeze(-1)], dim=-1)
    if next_token_id == tokenizer.eos_token_id:
        break

print("Input prompt:", input_text)
print("Generated text:", tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)[len(input_text):])  # remove the input text from the generated text
print("Test passed!")




