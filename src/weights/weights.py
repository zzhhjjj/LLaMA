from concurrent.futures import ThreadPoolExecutor
import os
from safetensors import safe_open
import torch
import pickle
from tqdm import tqdm


def load_weights_to_dict(path):
    """Load all the weights to the disk, return a dictionary containing all the tensors.

    Args:
        path: Directory containing the safetensors files.

    Returns:
        Dict: Dictionary containing all the tensors.
    """
    # Dictionary to store all tensors
    all_tensors = {}

    # Get a list of all files in the directory
    files = [f for f in os.listdir(path) if f.endswith(".safetensors")]

    # Iterate over all files in the directory with a progress bar
    for filename in tqdm(files, desc="Loading weights"):
        file_path = os.path.join(path, filename)
        with safe_open(file_path, framework="pt") as f:
            tensor_names = f.keys()
            for name in tensor_names:
                # Store each tensor in the dictionary
                all_tensors[name] = f.get_tensor(name)
    
    return all_tensors

def copy_tensor(tensor, destination):
    """Helper function to copy a tensor to the destination."""
    with torch.no_grad():
        destination.copy_(tensor)

def copy_weights_to_model(model, all_tensors):
    """Copy the weights from the dictionary to the model.

    Args:
        model: The model to which the weights are copied.
        all_tensors: Dictionary containing all the tensors.
    """
    # word embedding, lm head, final layer norm
    tasks = [
        (all_tensors['model.embed_tokens.weight'], model.word_embedding.weight),
        (all_tensors['lm_head.weight'], model.lm_head.weight),
        (all_tensors['model.norm.weight'], model.layer_norm.weight)
    ]
    
    is_merged_qkv_weight = os.getenv('MERGED_QKV_WEIGHT', '1')
    if is_merged_qkv_weight == '1':
        print(f"Using merged qkv weights: {is_merged_qkv_weight}")
    else:
        print("Using seperate qkv weights")
        
    num_layers = 32
    for i in range(num_layers):
        # LayerNorm
        tasks.append((all_tensors[f'model.layers.{i}.input_layernorm.weight'], model.layers[i].input_layernorm.weight))
        tasks.append((all_tensors[f'model.layers.{i}.post_attention_layernorm.weight'], model.layers[i].post_attention_layernorm.weight))
        
        # MLP
        tasks.append((all_tensors[f'model.layers.{i}.mlp.up_proj.weight'], model.layers[i].mlp.up_proj.weight))
        tasks.append((all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'], model.layers[i].mlp.gate_proj.weight))
        tasks.append((all_tensors[f'model.layers.{i}.mlp.down_proj.weight'], model.layers[i].mlp.down_proj.weight))
        
        # Attention
        if is_merged_qkv_weight == '1':
            ## merged qkv weights
            merged_qkv_weight = torch.cat([
                all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'],
                all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'],
                all_tensors[f'model.layers.{i}.self_attn.v_proj.weight']
            ], dim=0)
            tasks.append((merged_qkv_weight, model.layers[i].attention.qkv_proj.weight))
        else:
            ## Seperate qkv weights
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'], model.layers[i].attention.q_proj.weight))
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'], model.layers[i].attention.k_proj.weight))
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.v_proj.weight'], model.layers[i].attention.v_proj.weight))
            
        ## Output project
        tasks.append((all_tensors[f'model.layers.{i}.self_attn.o_proj.weight'], model.layers[i].attention.out_proj.weight))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda task: copy_tensor(*task), tasks), total=len(tasks), desc="Copying weights"))


# def copy_weights_to_model(model, all_tensors):
#     """Copy the weights from the dictionary to the model

#     Args:
#         model: The model to which the weights are copied
#         all_tensors: Dictionary containing all the tensors
#     """
#     # word embedding, lm head, final layer norm
#     with torch.no_grad():
#         model.word_embedding.weight.copy_(all_tensors['model.embed_tokens.weight']) 
#         model.lm_head.weight.copy_(all_tensors['lm_head.weight'])
#         model.layer_norm.weight.copy_(all_tensors['model.norm.weight'])
    
#         # decoder layers
#         num_layers = 32
#         for i in range(num_layers):
#             ## LayerNorm 
#             model.layers[i].input_layernorm.weight.copy_(all_tensors[f'model.layers.{i}.input_layernorm.weight'])
#             model.layers[i].post_attention_layernorm.weight.copy_(all_tensors[f'model.layers.{i}.post_attention_layernorm.weight'])
            
#             ## MLP 
#             model.layers[i].mlp.up_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.up_proj.weight'])
#             model.layers[i].mlp.gate_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'])
#             model.layers[i].mlp.down_proj.weight.copy_(all_tensors[f'model.layers.{i}.mlp.down_proj.weight'])
#             # # TODO: merge gate_proj and up_proj weights
#             # merged_gate_up_weight = torch.cat([all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'], all_tensors[f'model.layers.{i}.mlp.up_proj.weight']], dim=0)
#             # model.layers[i].mlp.gate_up_proj.weight.copy_(merged_gate_up_weight)
            
#             ## Attention
#             # merge qkv weights
#             merged_qkv_weight = torch.cat([all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'], all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'], all_tensors[f'model.layers.{i}.self_attn.v_proj.weight']], dim=0)
#             model.layers[i].attention.qkv_proj.weight.copy_(merged_qkv_weight)
#             model.layers[i].attention.out_proj.weight.copy_(all_tensors[f'model.layers.{i}.self_attn.o_proj.weight'])
        