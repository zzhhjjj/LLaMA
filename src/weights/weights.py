import os
from safetensors import safe_open
import torch
import pickle


weights_directory_path = "/fsx/haojun/LLaMA/.cache/models/Meta-Llama-3-8B"
def load_weights_to_dict(path):
    """load all the weights to the disk, return a dictionary containing all the tensors

    Args:
        path: Directory containing the safetensors files

    Returns:
        Dict: Dictionary containing all the tensors
    """
    # Dictionary to store all tensors
    all_tensors = {}

    # Iterate over all files in the directory
    for filename in os.listdir(weights_directory_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(weights_directory_path, filename)
            with safe_open(file_path, framework="pt") as f:
                tensor_names = f.keys()
                for name in tensor_names:
                    # Store each tensor in the dictionary
                    all_tensors[name] = f.get_tensor(name)
    print(all_tensors.keys())
    return all_tensors

def copy_weights_to_model(model, all_tensors):
    """Copy the weights from the dictionary to the model

    Args:
        model: The model to which the weights are copied
        all_tensors: Dictionary containing all the tensors
    """
    # word embedding, lm head, final layer norm
    with torch.no_grad():
        model.word_embedding.weight.copy_(all_tensors['model.embed_tokens.weight']) 
        model.lm_head.weight.copy_(all_tensors['lm_head.weight'])
        model.layer_norm.weight.copy_(all_tensors['model.norm.weight'])
    
    # decoder layers
    num_layers = 32
    with torch.no_grad():
        for i in range(num_layers):
            ## LayerNorm 
            model.layers[i].input_layernorm.weight.copy_(all_tensors[f'model.layers.{i}.input_layernorm.weight'])
            model.layers[i].post_attention_layernorm.weight.copy_(all_tensors[f'model.layers.{i}.post_attention_layernorm.weight'])
            
            ## MLP 
            model.layers[i].mlp.up_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.up_proj.weight'])
            model.layers[i].mlp.gate_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'])
            model.layers[i].mlp.down_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.down_proj.weight'])
            # # TODO: merge gate_proj and up_proj weights
            # merged_gate_up_weight = torch.cat([all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'], all_tensors[f'model.layers.{i}.mlp.up_proj.weight']], dim=0)
            # model.layers[i].mlp.gate_up_proj.weight.copy_(merged_gate_up_weight)
            
            ## Attention
            # merge qkv weights
            merged_qkv_weight = torch.cat([all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'], all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'], all_tensors[f'model.layers.{i}.self_attn.v_proj.weight']], dim=0)
            model.layers[i].attention.qkv_proj.weight.copy_(merged_qkv_weight)
            model.layers[i].attention.out_proj.weight.copy_(all_tensors[f'model.layers.{i}.self_attn.o_proj.weight'])
        