"""
This script compare the output logits/generated token of my LLaMA model with the transformers LLaMMA model.
"""

import sys
from tools.debug.debug_llama import DebugMyCausalSelfAttentionforward, DebugMyDecoderLayerforward, DebugMyLLaMAforward
from src.model.llama3 import LLaMA
from tools.debug.replace_attn import DebugDecoderLayerforward, DebugRoPEforward, DebugSDPAforward
import torch 
from dataclasses import dataclass
from src.weights.weights import load_weights_to_dict, copy_weights_to_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tools.debug.logs import RedirectOutput, get_log_file_path, print_env_variables, timer
import lovely_tensors as lt
lt.monkey_patch()
from lovely_tensors import set_config
set_config(precision=6)

compare_activation_value = False # whether to compare each activation value(assert_close). Otherwise only compare the output logits.
assert_equal = True # whether to assert each activation value is the same(torch.eq).

# to match the output of transformers model, set MERGED_QKV_WEIGHT to 0 is necessary
os.environ['DATA_TYPE'] = 'bfloat16' # bfloat16/float32
os.environ['MERGED_QKV_WEIGHT'] = '0' # 1/0
os.environ['MERGED_GATE_UP_WEIGHT'] = '1' # 1/0
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

with timer('Load LLaMA model'):
    ## initialize model 
    config = LLaMAConfig()
    model = LLaMA(config)

    ## load weights 
    weights_directory_path = ".cache/models/Meta-Llama-3-8B"
    all_tensors = load_weights_to_dict(weights_directory_path)
    copy_weights_to_model(model, all_tensors)
    model = model.to(dtype).to(device)

## Tokenizer
pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

## Reference model
with timer("Load transformers model"):
    attn_implementation = 'sdpa' if os.getenv('ATTENTION', 'SDPA') == 'SDPA' else 'flash_attention_2'
    transformer_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype, attn_implementation = attn_implementation).to(device)

if compare_activation_value:
    ## Replace the forward function of the transformer model. Save the activations
    for i, layer in enumerate(transformer_model.model.layers):
        layer.layer_idx = i
        layer.forward = DebugDecoderLayerforward.__get__(layer, type(layer))
        if os.getenv('ATTENTION', 'SDPA') == 'SDPA': # comparaison for Flash attention is not implemented yet
            layer.self_attn.forward = DebugSDPAforward.__get__(layer.self_attn, type(layer.self_attn))
        layer.self_attn.rotary_emb.layer_idx = i
        layer.self_attn.rotary_emb.forward = DebugRoPEforward.__get__(layer.self_attn.rotary_emb, type(layer.self_attn.rotary_emb))
    
    # which decoder layers to compare
    debug_arguments = {'layers': [i for i in range(10)], 'variables': [1,2,3,4]}
    
    ## Do the same for my model
    model.forward = DebugMyLLaMAforward.__get__(model, type(model))
    for i,layer in enumerate(model.layers):
        layer.debug_arguments = debug_arguments
        layer.assert_equal = False
        layer.forward = DebugMyDecoderLayerforward.__get__(layer, type(layer))
        if os.getenv('ATTENTION', 'SDPA') == 'SDPA': # comparaison for Flash attention is not implemented yet
            layer.attention.debug_arguments = debug_arguments
            layer.attention.assert_equal = False
            layer.attention.forward = DebugMyCausalSelfAttentionforward.__get__(layer.attention, type(layer.attention))

## prompt
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
max_new_tokens = 100

log_file_path = get_log_file_path() # depends on the env variables

with RedirectOutput(log_file_path):    # empty string for terminal output
    # check environment variables
    print_env_variables()
    
    # generate text
    for i in range(max_new_tokens):
        output_logits = transformer_model(**inputs)['logits'].to(dtype) # output logits of transformer model is still in float32 even dtype is bfloat16
        my_output_logits = model(**inputs)
        next_token_id = torch.argmax(output_logits[:, -1, :], dim=-1)
        my_next_token_id = torch.argmax(my_output_logits[:, -1, :], dim=-1)
        try:
            # test logits and generation on the same time.
            torch.testing.assert_close(my_output_logits,output_logits,rtol=1e-5,atol=1e-5) # check if the output logits are close
            assert torch.equal(my_output_logits,output_logits), "Output logits are not the same"
        except AssertionError as e:
            print(f'Token {i+1} failed: {e}')
            print("Reference: ",output_logits) 
            print("My output: ",my_output_logits)
        assert next_token_id == my_next_token_id, f"Predictions are not the same: {next_token_id} != {my_next_token_id}"
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id.unsqueeze(-1)], dim=-1)
        if next_token_id == tokenizer.eos_token_id:
            break
    print("Input prompt:", input_text)
    print("Generated text:", tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)[len(input_text):])  # remove the input text from the generated text
    print("Test passed!")