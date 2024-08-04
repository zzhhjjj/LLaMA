import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import lovely_tensors as lt; lt.monkey_patch()
from lovely_tensors import set_config
set_config(precision=6)
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb, LlamaSdpaAttention, LlamaDecoderLayer
import torch.nn as nn
from tqdm import tqdm

# override the forward function of LlamaSdpaAttention
def DebugDecoderLayerforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    # add debug code here
    debug_arguments = {
        'layers': [i for i in range(10)], # layers to print
        'variables': [1,2,3,4], # variables to print
    }
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
    os.makedirs(folder_to_save, exist_ok=True)
        
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    
    layers = debug_arguments['layers']
    variables = debug_arguments['variables']
    if self.layer_idx in layers:
        print('Layer: ', self.layer_idx)
        print("Input hidden states:", residual)
        print("Hidden states after LN:", hidden_states)
        
    if self.layer_idx in layers:
        torch.save(residual.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_input_hidden_states.pt'))
        torch.save(hidden_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_hidden_states_after_ln.pt'))
        
    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    hidden_states = residual + hidden_states

    if self.layer_idx in layers:
        print('Residual + Hidden : ', hidden_states)
        print("After post attention LN:", self.post_attention_layernorm(hidden_states))
        print('MLP output:', self.mlp(self.post_attention_layernorm(hidden_states)))
    if self.layer_idx in layers:
        torch.save(hidden_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_residual_plus_hidden.pt'))
        torch.save(self.post_attention_layernorm(hidden_states).cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_post_attention_ln.pt'))
        torch.save(self.mlp(self.post_attention_layernorm(hidden_states)).cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_mlp_output.pt'))

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def DebugSDPAforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # which layers and variables to print
    debug_arguments = {
        'layers': [i for i in range(10)], # layers to print
        'variables': [1,2,3,4], # variables to print
    }
        
    bsz, q_len, _ = hidden_states.size()
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
    os.makedirs(folder_to_save, exist_ok=True)
    layers = debug_arguments['layers']
    variables = debug_arguments['variables']
    
    if self.layer_idx in layers and 2 in variables:
        print("Q reshaped:", query_states)
        print("K reshaped:", key_states)
        print("V reshaped:", value_states)
    if self.layer_idx in layers and 2 in variables:
        torch.save(query_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_query_reshaped.pt'))
        torch.save(key_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_key_reshaped.pt'))
        torch.save(value_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_value_reshaped.pt'))

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if self.layer_idx in layers and 3 in variables:
        print("Q after rotary pos emb:", query_states)
        print("K after rotary pos emb:", key_states)
    if self.layer_idx in layers and 3 in variables:
        torch.save(query_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_query_after_rotary.pt'))
        torch.save(key_states.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_key_after_rotary.pt'))
    
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    if self.layer_idx in layers:
        print("SDPA output:", attn_output)
    if self.layer_idx in layers:
        torch.save(attn_output.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_sdpa_output.pt'))

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if self.layer_idx in layers:  
        print("Self Attention output:", attn_output)
    if self.layer_idx in layers:
        torch.save(attn_output.cpu(), os.path.join(folder_to_save, f'layer_{self.layer_idx}_self_attention_output.pt'))

    return attn_output, None, past_key_value

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim%2==0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
    position = torch.arange(seq_length).unsqueeze(1).to(device).float() # [seq_length, 1]
    return torch.cos(position.float()*theta.float()).repeat(1,2), torch.sin(position.float()*theta.float()).repeat(1,2) # [seq_length, head_dim], [seq_length, head_dim]

# The code looks messy. However, the conclusion is that the computing device will affect the result.
@torch.no_grad()
def DebugRoPEforward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)
        
    import os 
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
    saved_my_theta = torch.load(os.path.join(folder_to_save, f'my_theta.pt'))
    saved_inv_freq = torch.load(os.path.join(folder_to_save, f'inv_freq.pt'))
    
    # saved_my_theta != my theta lol. WTF ! 
    base = 500000.0
    head_dim = x.size(-1)
    seq_length = position_ids.size(1)
    device = x.device
    my_theta_cpu = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim)) # =  self.inv_freq.cpu() 
    my_theta_gpu = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cuda') / head_dim)) # != self.inv_freq
    
    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling
    
    my_position = torch.arange(seq_length).unsqueeze(1).to(device).float() # [seq_length, 1]
    my_cos, my_sin = torch.cos(my_position.float()* my_theta_cpu.to(device).float()).repeat(1,2), torch.sin(my_position.float()*my_theta_cpu.to(device).float()).repeat(1,2) #  = cos, sin
    my_cos, my_sin = torch.cos(my_position.to('cpu').float()* my_theta_cpu.float()).repeat(1,2), torch.sin(my_position.to('cpu').float()*my_theta_cpu.float()).repeat(1,2) # != cos, sin
    
    if self.layer_idx == 0: 
        print("RoPE cos:", cos)
        print("RoPE sin:", sin)
        folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
        torch.save(cos.cpu(), os.path.join(folder_to_save, f'cos.pt'))
        saved_cos = torch.load(os.path.join(folder_to_save, f'cos.pt'))
        assert torch.equal(cos.cpu(), saved_cos), f"Save cos error."
        torch.save(sin.cpu(), os.path.join(folder_to_save, f'sin.pt'))

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



# class CustomDebugLlamaSdpaAttention(LlamaSdpaAttention):
#     """
#     Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     LlamaAttention as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
#     SDPA API.
#     """
#     def __init__(self, config: LlamaConfig, layer_idx: int, debug_arguments) -> None:
#         super().__init__(config)
#         self.layer_idx = layer_idx
#         self.debug_arguments = debug_arguments

#     # Adapted from LlamaAttention.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if output_attentions:
#             # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
#             print(
#                 "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
#                 'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return super().forward(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#             )
#         # which layers and variables to print
#         layers = self.debug_arguments['layers']
#         variables = self.debug_arguments['variables']
        
#         if self.layer_idx in layers:
#             print('Layer: ', self.layer_idx)
            
#         if self.layer_idx in layers and 1 in variables:  
#             print("Hidden state after LN:", hidden_states)
            
#         bsz, q_len, _ = hidden_states.size()
        
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)
        
        
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
#         if self.layer_idx in layers and 2 in variables:
#             print("Q reshaped:", query_states)
#             print("K reshaped:", key_states)
#             print("V reshaped:", value_states)

#         cos, sin = self.rotary_emb(value_states, position_ids)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#         if self.layer_idx in layers and 3 in variables:
#             print("Q after rotary pos emb:", query_states)
#             print("K after rotary pos emb:", key_states)
        
#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         causal_mask = attention_mask
#         if attention_mask is not None:
#             causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

#         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#         # Reference: https://github.com/pytorch/pytorch/issues/112577.
#         if query_states.device.type == "cuda" and causal_mask is not None:
#             query_states = query_states.contiguous()
#             key_states = key_states.contiguous()
#             value_states = value_states.contiguous()

#         # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
#         # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
#         is_causal = True if causal_mask is None and q_len > 1 else False

#         attn_output = torch.nn.functional.scaled_dot_product_attention(
#             query_states,
#             key_states,
#             value_states,
#             attn_mask=causal_mask,
#             dropout_p=self.attention_dropout if self.training else 0.0,
#             is_causal=is_causal,
#         )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         if self.layer_idx in layers and 4 in variables:
#             print("Attention output reshaped:", attn_output)
            
#         attn_output = attn_output.view(bsz, q_len, -1)

#         attn_output = self.o_proj(attn_output)

#         return attn_output, None, past_key_value


# class CustomDebugLlamaDecoderLayer(nn.Module):
#     def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
#         super().__init__()
#         self.layer_idx = layer_idx
        
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
#         # add debug code here
#         debug_arguments = {
#             'layers': [1,2,3], # layers to print
#             'variables': [1,2,3,4], # variables to print
#         }
#         layers = debug_arguments['layers']
#         variables = debug_arguments['variables']
        
#         if self.layer_idx in layers:
#             print('Layer: ', self.layer_idx)
            
#         if self.layer_idx in layers and 1 in variables:  
#             print("Input hidden states:", hidden_states)
            
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)
        
#         if self.layer_idx in layers and 1 in variables:  
#             print("hidden states after LN:", hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         if self.layer_idx in layers and 1 in variables:  
#             print("Attention output:", hidden_states)

#         hidden_states = residual + hidden_states

#         if self.layer_idx in layers:
#             print('Residual + Hidden : ', hidden_states)
#             print("After post attention LN:", self.post_attention_layernorm(hidden_states))
#             print('MLP output:', self.mlp(self.post_attention_layernorm(hidden_states)))

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs





