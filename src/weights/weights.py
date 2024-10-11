from concurrent.futures import ThreadPoolExecutor
import glob
import os
from safetensors import safe_open
from safetensors.torch import save_file
from src.parallel.initialize import get_data_parallel_rank, get_data_parallel_world_size, get_model_parallel_rank, get_model_parallel_world_size
import torch
import pickle
from tqdm import tqdm
import torch.distributed as dist


# Functions for load from transformer model
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

    # Function to load a single file
    def load_file(filename):
        file_path = os.path.join(path, filename)
        with safe_open(file_path, framework="pt") as f:
            return {name: f.get_tensor(name) for name in f.keys()}

    # Load all files concurrently
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_file, files), total=len(files), desc="Loading weights"))
    
    # Combine all results into a single dictionary
    for result in results:
        all_tensors.update(result)
    
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
   
    num_layers = 32
    for i in range(num_layers):
        # LayerNorm
        tasks.append((all_tensors[f'model.layers.{i}.input_layernorm.weight'], model.layers[i].input_layernorm.weight))
        tasks.append((all_tensors[f'model.layers.{i}.post_attention_layernorm.weight'], model.layers[i].post_attention_layernorm.weight))
        
        # MLP
        if os.getenv('MERGED_GATE_UP_WEIGHT', '1') == '1':
            # merge gate and up projection weights
            merged_gate_up_weight = torch.cat([
                all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'],
                all_tensors[f'model.layers.{i}.mlp.up_proj.weight']
            ], dim=0)
            tasks.append((merged_gate_up_weight, model.layers[i].mlp.gate_up_proj.weight))
        else:
            tasks.append((all_tensors[f'model.layers.{i}.mlp.up_proj.weight'], model.layers[i].mlp.up_proj.weight))
            tasks.append((all_tensors[f'model.layers.{i}.mlp.gate_proj.weight'], model.layers[i].mlp.gate_proj.weight))
        tasks.append((all_tensors[f'model.layers.{i}.mlp.down_proj.weight'], model.layers[i].mlp.down_proj.weight))
        
        # Attention
        if os.getenv('MERGED_QKV_WEIGHT', '1') == '1':
            # merged qkv weights
            merged_qkv_weight = torch.cat([
                all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'],
                all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'],
                all_tensors[f'model.layers.{i}.self_attn.v_proj.weight']
            ], dim=0)
            tasks.append((merged_qkv_weight, model.layers[i].attention.qkv_proj.weight))
        else:
            # Seperate qkv weights
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.q_proj.weight'], model.layers[i].attention.q_proj.weight))
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.k_proj.weight'], model.layers[i].attention.k_proj.weight))
            tasks.append((all_tensors[f'model.layers.{i}.self_attn.v_proj.weight'], model.layers[i].attention.v_proj.weight))
            
        ## Output project
        tasks.append((all_tensors[f'model.layers.{i}.self_attn.o_proj.weight'], model.layers[i].attention.out_proj.weight))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda task: copy_tensor(*task), tasks), total=len(tasks), desc="Copying weights"))


## Function for save and load weights below
# save: 
# 1. Let each process save its own weights
# 2. Merge shared weights (will do it later, by using the parameter names)
#    For now just suppose TP will not change, so each model load the corresponding TP weights
# load:
# 1. let each process load its own weights

def save_weights(model, file_prefix):
    """
    Save all the weights to the disk.
    """
    # hardcode the parameter names 
    column_parallel_params = ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'gate_proj', 'word_embedding', 'lm_head'] # split on the 0th dimension
    row_parallel_params = ['out_proj', 'down_proj'] # split on the 1st dimension
    
    # Get tensor parallel rank and world size
    tp_rank = get_model_parallel_rank()
    tp_world_size = get_model_parallel_world_size()
    dp_rank = get_data_parallel_rank()
    dp_world_size = get_data_parallel_world_size()
    
    if dp_world_size > 1:
        model = model.module # unwarp from DDP
    
    # DP rank = 0 saves the sharded weights()
    # DP rank = 0 + TP rank = 0 saves the non-sharded weights
    if dp_rank == 0:
        # Dictionary to store this rank's parameters
        unsharded_state_dict = {}
        sharded_state_dict = {}
        for param_name, param in model.named_parameters():
            if any(name in param_name for name in column_parallel_params) or any(name in param_name for name in row_parallel_params):
                sharded_state_dict[param_name] = param.data.cpu()
                # print(f"Sharded: {param_name}")
            else:
                if tp_rank == 0:
                    unsharded_state_dict[param_name] = param.data.cpu()
                    # print(f"Non-Parallel: {param_name}")

        # Ensure the directory exists
        directory = os.path.dirname(file_prefix)
        os.makedirs(directory, exist_ok=True)
            
        shared_file_name = f"{file_prefix}sharded_tp_rank_{tp_rank}_tp_world_size_{tp_world_size}.safetensors"
        save_file(sharded_state_dict, shared_file_name)
        
        if tp_rank == 0:
            unsharded_file_name = f"{file_prefix}unsharded.safetensors"
            save_file(unsharded_state_dict, unsharded_file_name)
        

def load_weights(model, file_prefix):
    """
    Load the weights from the disk for the current TP rank.
    Each TP rank loads the sharded weights relevant to its rank.
    """
    # two type of directory: 
    # 1. weights under the file_prefix
    # 2. weights under the file_prefix/epochs (not implemented yet)
    if not glob.glob(os.path.join(file_prefix, "*.safetensors")):
        if dist.get_rank() == 0:
            print(f"No weights found in {file_prefix} Skipping weight loading.")
        return
    if dist.get_rank() == 0:
        print(f"Starting to load weights.")
    # Get tensor parallel rank and world size
    tp_rank = get_model_parallel_rank()
    tp_world_size = get_model_parallel_world_size()

    # Dictionary to hold the parameters to be loaded for this rank
    loaded_state_dict = {}

    # Load the sharded (shared) weights for this TP rank
    sharded_file = f"{file_prefix}sharded_tp_rank_{tp_rank}_tp_world_size_{tp_world_size}.safetensors"
    print(f"Loading sharded weights for TP rank {tp_rank} from {sharded_file}")
    
    assert os.path.exists(sharded_file), f"File {sharded_file} does not exist. For now, we assume that TP degree is fixed."
    with safe_open(sharded_file, framework="pt") as f:
        for param_name in f.keys():
            loaded_state_dict[param_name] = f.get_tensor(param_name)

        
    unsharded_file = f"{file_prefix}unsharded.safetensors"
    with safe_open(unsharded_file, framework="pt") as f:
        for param_name in f.keys():
            loaded_state_dict[param_name] = f.get_tensor(param_name)

    # Load the weights into the model's state_dict
    model.load_state_dict(loaded_state_dict, strict=False)
    print(f"Weights successfully loaded for TP rank {tp_rank}.")


            
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
        