"""
Just copy the model code and add code printing, loading, test to debug.
"""
import os
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from src.model.llama3 import apply_rotary_pos_emb, flash_attention
import lovely_tensors as lt
from lovely_tensors import set_config
lt.monkey_patch()
set_config(precision=6)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32


def DebugMyCausalSelfAttentionforward(self, x, cos, sin, attention_mask=None):
    assert self.debug_arguments!=None and 'layers' in self.debug_arguments and 'variables' in self.debug_arguments
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
    os.makedirs(folder_to_save, exist_ok=True)
    layers = self.debug_arguments['layers']
    variables = self.debug_arguments['variables']
    
    batch_size, seq_length, hidden_dim = x.size()
    if self.is_merged_qkv_weight == '1':
        qkv = self.qkv_proj(x) # [batch_size, seq_length, num_heads*head_dim + 2*num_key_values*head_dim]
        q, k, v = torch.split(qkv,
            [
                self.num_heads * self.head_dim,
                self.num_key_values * self.head_dim,
                self.num_key_values * self.head_dim,
            ],
            dim=-1,
        )
    else:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

    q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_length, head_dim]
    k = k.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
    v = v.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
    
    if self.layer_idx in layers and 2 in variables:
        # print("Q reshaped:", q)
        # print("K reshaped:", k)
        # print("V reshaped:", v)
        
        saved_q = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_query_reshaped.pt'))
        saved_k = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_key_reshaped.pt'))
        saved_v = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_value_reshaped.pt'))
        
        try:
            torch.testing.assert_close(q.cpu(), saved_q, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(k.cpu(), saved_k, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(v.cpu(), saved_v, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(q.cpu(), saved_q), f"Layer {self.layer_idx}: Query reshaped mismatch."
                assert torch.equal(k.cpu(), saved_k), f"Layer {self.layer_idx}: Key reshaped mismatch."
                assert torch.equal(v.cpu(), saved_v), f"Layer {self.layer_idx}: Value reshaped mismatch."
        except AssertionError as e:
            print("Layer", self.layer_idx, "Query/Key/Value reshaped mismatch.")
            print(e)

    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    
    if self.layer_idx in layers and 3 in variables:
        # print("Q after rotary pos emb:", q)
        # print("K after rotary pos emb:", k)

        saved_q_after_rotary = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_query_after_rotary.pt'))
        saved_k_after_rotary = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_key_after_rotary.pt'))
        
        try:
            torch.testing.assert_close(q.cpu(), saved_q_after_rotary, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(k.cpu(), saved_k_after_rotary, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(q.cpu(), saved_q_after_rotary), f"Layer {self.layer_idx}: Query after rotary pos emb mismatch."
                assert torch.equal(k.cpu(), saved_k_after_rotary), f"Layer {self.layer_idx}: Key after rotary pos emb mismatch."
        except AssertionError as e:
            print("Layer", self.layer_idx, "Query/Key after rotary pos emb mismatch.")
            print(e)

    k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
    v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
   
    if os.getenv('ATTENTION', 'SDPA') == 'SDPA':
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2) # [batch_size, seq_length, num_heads, head_dim]
    else:
        out = flash_attention(q, k, v) # [batch_size, seq_length, num_heads, head_dim] 
    if self.layer_idx in layers:
        # print("SDPA output:", out)
        saved_out = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_sdpa_output.pt')).transpose(1, 2)
        try:
            torch.testing.assert_close(out.cpu(), saved_out, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(out.cpu(), saved_out), f"Layer {self.layer_idx}: SDPA/Flash output mismatch."
        except AssertionError as e:
            print("Layer", self.layer_idx, "SDPA/Flash output mismatch.")
            print(e)

    out = out.reshape(batch_size, seq_length, self.num_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
    out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
    
    if self.layer_idx in layers:
        saved_self_attention_output = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_self_attention_output.pt'))
        try:
            torch.testing.assert_close(out.cpu(), saved_self_attention_output, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(out.cpu(), saved_self_attention_output), f"Layer {self.layer_idx}: Self attention output mismatch."
        except AssertionError as e:
            print("Layer", self.layer_idx, "Self attention output mismatch.")
            print(e)
        # print("Self Attention output:", out)
    
    return out

def DebugMyDecoderLayerforward(self, x, cos, sin, attention_mask = None):
    layers = self.debug_arguments['layers']
    variables = self.debug_arguments['variables']
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'

    if self.layer_idx in layers:
        # print('Layer: ', self.layer_idx)
        # print("Input hidden states:", x)
        # print("Hidden states after LN:", self.input_layernorm(x))
        saved_input_hidden_states = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_input_hidden_states.pt'))
        saved_hidden_states_after_ln = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_hidden_states_after_ln.pt'))
        
        try:
            torch.testing.assert_close(x.cpu(), saved_input_hidden_states, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.input_layernorm(x).cpu(), saved_hidden_states_after_ln, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(x.cpu(), saved_input_hidden_states), f"Layer {self.layer_idx}: Input hidden states mismatch."
                assert torch.equal(self.input_layernorm(x).cpu(), saved_hidden_states_after_ln), f"Layer {self.layer_idx}: Hidden states after LN mismatch."
        except:
            print("Layer", self.layer_idx, "Input hidden states mismatch.")
            print("Layer", self.layer_idx, "Hidden states after LN mismatch.")

    x = x + self.attention(self.input_layernorm(x),cos,sin)
    if self.layer_idx in layers:
        # print('Residual + Hidden : ', x)
        # print("After post attention LN:", self.post_attention_layernorm(x))
        # print('MLP output:', self.mlp(self.post_attention_layernorm(x)))
        
        saved_residual_plus_hidden = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_residual_plus_hidden.pt'))
        saved_post_attention_ln = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_post_attention_ln.pt'))
        saved_mlp_output = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_mlp_output.pt'))
        
        try:
            torch.testing.assert_close(x.cpu(), saved_residual_plus_hidden, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.post_attention_layernorm(x).cpu(), saved_post_attention_ln, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.mlp(self.post_attention_layernorm(x)).cpu(), saved_mlp_output, rtol=1e-5, atol=1e-5)
            if self.assert_equal:
                assert torch.equal(x.cpu(), saved_residual_plus_hidden), f"Layer {self.layer_idx}: Residual plus hidden states mismatch."
                assert torch.equal(self.post_attention_layernorm(x).cpu(), saved_post_attention_ln), f"Layer {self.layer_idx}: Post attention LN mismatch."
                assert torch.equal(self.mlp(self.post_attention_layernorm(x)).cpu(), saved_mlp_output), f"Layer {self.layer_idx}: MLP output mismatch."
        except AssertionError as e:
            # print("Layer", self.layer_idx, "Residual plus hidden states mismatch.")
            # print("Layer", self.layer_idx, "Post attention LN mismatch.")
            # print("Layer", self.layer_idx, "MLP output mismatch.")
            print("Layer", self.layer_idx)
            print(e)
    x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
    return x


def DebugMyLLaMAforward(self, input_ids, attention_mask):
    batch_size, seq_length = input_ids.size()
    x = self.word_embedding(input_ids)

    cos, sin = self.cos[:seq_length], self.sin[:seq_length]
    folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
    
    # print("RoPE cos:", cos)
    # print("RoPE sin:", sin)
    saved_cos = torch.load(os.path.join(folder_to_save, f'cos.pt'))
    saved_sin = torch.load(os.path.join(folder_to_save, f'sin.pt'))
    saved_cos = saved_cos.squeeze(0)
    saved_sin = saved_sin.squeeze(0)
    torch.testing.assert_close(cos.cpu(), saved_cos,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(sin.cpu(), saved_sin,rtol=1e-5,atol=1e-5)
    assert torch.equal(cos.cpu(), saved_cos), f"RoPE cos mismatch."
    assert torch.equal(sin.cpu(), saved_sin), f"RoPE sin mismatch."

    for i, layer in enumerate(self.layers):
        x = layer(x, cos, sin, attention_mask)  # [batch_size, seq_length, hidden_dim]

    x = self.layer_norm(x)
    # print("After Final Layer Norm:", x)

    logits = self.lm_head(x)
    # print("Output Logits:", logits)

    return logits  # [batch_size, seq_length, vocab_size]
