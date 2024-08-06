"""
Just copy the model code and add code printing, loading, test to debug.
"""
import os
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
import lovely_tensors as lt
from lovely_tensors import set_config
lt.monkey_patch()
set_config(precision=6)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32


def apply_rotary_pos_emb(x, cos, sin):
    batch_size, num_head, seq_length, head_dim = x.size()
    assert cos.size(0)==seq_length
    assert cos.size(1)==head_dim
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x = x * cos + rotate_half * sin
    return x

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim%2==0
    # Results on CUDA and CPU are different even with the same formula
    # To match transformers implementation. frequency should be computed on CPU
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    position = torch.arange(seq_length).unsqueeze(1).to(device).float() # [seq_length, 1]
    # To match transformers implementation. m * theta should be computed on GPU
    theta = theta.to(device)
    return torch.cos(position.float()*theta.float()).to(dtype).repeat(1,2), torch.sin(position.float()*theta.float()).to(dtype).repeat(1,2) # [seq_length, head_dim], [seq_length, head_dim]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def flash_attention(q, k, v):
    q = q.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    k = k.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    v = v.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    return flash_attn_func(q, k, v, causal=True)

def scaled_dot_product_attention(q,k,v,is_causal=True,mask=None):
    # input: q,k,v of shape (batch_size, num_head, seq_length, head_dim) 
    # output: output of shape (batch_size, seq_length, head_dim*num_head)
    assert len(q.shape)==4 
    assert is_causal==True or mask!=None
    batch_size, num_head, seq_length, head_dim = q.shape
    if is_causal:
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(device)
    scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1)))
    scaled_dot_product.masked_fill_(mask, float('-inf'))
    attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1) # [batch_size, num_head, seq_length, seq_length]
    return torch.matmul(attention_weights, v).transpose(1,2).reshape(batch_size, seq_length, num_head*head_dim).contiguous() # [batch_size, seq_length, head_dim*num_head]

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_key_values = config.num_key_values
        self.head_dim = self.hidden_dim//self.num_heads
        self.is_merged_qkv_weight = os.getenv('MERGED_QKV_WEIGHT', '1')
        if self.is_merged_qkv_weight  == '1': 
            self.qkv_proj = nn.Linear(config.hidden_dim, self.num_heads*self.head_dim + 2*self.num_key_values*self.head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.hidden_dim, self.num_heads*self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layer_idx = layer_idx
        
        ## TODO support mask
    
    def forward(self, x, cos, sin, attention_mask=None, debug_arguments=None):
        assert debug_arguments!=None and 'layers' in debug_arguments and 'variables' in debug_arguments
        folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
        os.makedirs(folder_to_save, exist_ok=True)
        layers = debug_arguments['layers']
        variables = debug_arguments['variables']
        
        rtol = 1e-5
        atol = 1e-5
        
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
            
            torch.testing.assert_close(q.cpu(), saved_q, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(k.cpu(), saved_k, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(v.cpu(), saved_v, rtol=1e-5, atol=1e-5)

            assert torch.equal(q.cpu(), saved_q), f"Layer {self.layer_idx}: Query reshaped mismatch."
            assert torch.equal(k.cpu(), saved_k), f"Layer {self.layer_idx}: Key reshaped mismatch."
            assert torch.equal(v.cpu(), saved_v), f"Layer {self.layer_idx}: Value reshaped mismatch."

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        if self.layer_idx in layers and 3 in variables:
            # print("Q after rotary pos emb:", q)
            # print("K after rotary pos emb:", k)

            saved_q_after_rotary = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_query_after_rotary.pt'))
            saved_k_after_rotary = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_key_after_rotary.pt'))
            
            torch.testing.assert_close(q.cpu(), saved_q_after_rotary, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(k.cpu(), saved_k_after_rotary, rtol=1e-5, atol=1e-5)

            assert torch.equal(q.cpu(), saved_q_after_rotary), f"Layer {self.layer_idx}: Query after rotary pos emb mismatch."
            assert torch.equal(k.cpu(), saved_k_after_rotary), f"Layer {self.layer_idx}: Key after rotary pos emb mismatch."

        # k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        # v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        k = repeat_kv(k,self.num_heads // self.num_key_values)
        v = repeat_kv(v,self.num_heads // self.num_key_values)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, num_heads, seq_length, head_dim]
        if self.layer_idx in layers:
            # print("SDPA output:", out)
            saved_out = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_sdpa_output.pt'))
            torch.testing.assert_close(out.cpu(), saved_out, rtol=1e-5, atol=1e-5)
            assert torch.equal(out.cpu(), saved_out), f"Layer {self.layer_idx}: SDPA output mismatch."
            
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.num_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        
        if self.layer_idx in layers:
            saved_self_attention_output = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_self_attention_output.pt'))
            torch.testing.assert_close(out.cpu(), saved_self_attention_output, rtol=1e-5, atol=1e-5)
            assert torch.equal(out.cpu(), saved_self_attention_output), f"Layer {self.layer_idx}: Self attention output mismatch."
            # print("Self Attention output:", out)
        
        return out

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LLaMAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.merged_gate_up = os.getenv('MERGED_GATE_UP_WEIGHT', '1') == '1'
        if self.merged_gate_up:
            print("Using merged gate and up projection weights")
            self.gate_up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim*2, bias=False)
        else:
            self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        
    def forward(self, x):
        if  self.merged_gate_up:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_dim)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_dim)
        self.attention = CausalSelfAttention(config, layer_idx = layer_idx)
        self.mlp = LLaMAMLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin, attention_mask = None, debug_arguments = None):
        layers = debug_arguments['layers']
        variables = debug_arguments['variables']
        folder_to_save = '/fsx/haojun/LLaMA/.cache/activation_values'
        
        # rtol = 1e-7
        # atol = 1e-7
        
        if self.layer_idx in layers:
            # print('Layer: ', self.layer_idx)
            # print("Input hidden states:", x)
            # print("Hidden states after LN:", self.input_layernorm(x))
            saved_input_hidden_states = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_input_hidden_states.pt'))
            saved_hidden_states_after_ln = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_hidden_states_after_ln.pt'))
            
            torch.testing.assert_close(x.cpu(), saved_input_hidden_states, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.input_layernorm(x).cpu(), saved_hidden_states_after_ln, rtol=1e-5, atol=1e-5)

            assert torch.equal(x.cpu(), saved_input_hidden_states), f"Layer {self.layer_idx}: Input hidden states mismatch."
            assert torch.equal(self.input_layernorm(x).cpu(), saved_hidden_states_after_ln), f"Layer {self.layer_idx}: Hidden states after LN mismatch."

        x = x + self.attention(self.input_layernorm(x),cos,sin, debug_arguments = debug_arguments)
        if self.layer_idx in layers:
            # print('Residual + Hidden : ', x)
            # print("After post attention LN:", self.post_attention_layernorm(x))
            # print('MLP output:', self.mlp(self.post_attention_layernorm(x)))
            
            saved_residual_plus_hidden = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_residual_plus_hidden.pt'))
            saved_post_attention_ln = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_post_attention_ln.pt'))
            saved_mlp_output = torch.load(os.path.join(folder_to_save, f'layer_{self.layer_idx}_mlp_output.pt'))
            
            torch.testing.assert_close(x.cpu(), saved_residual_plus_hidden, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.post_attention_layernorm(x).cpu(), saved_post_attention_ln, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(self.mlp(self.post_attention_layernorm(x)).cpu(), saved_mlp_output, rtol=1e-5, atol=1e-5)

            assert torch.equal(x.cpu(), saved_residual_plus_hidden), f"Layer {self.layer_idx}: Residual plus hidden states mismatch."
            assert torch.equal(self.post_attention_layernorm(x).cpu(), saved_post_attention_ln), f"Layer {self.layer_idx}: Post attention LN mismatch."
            assert torch.equal(self.mlp(self.post_attention_layernorm(x)).cpu(), saved_mlp_output), f"Layer {self.layer_idx}: MLP output mismatch."
            
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x
    
class DebugLLaMA(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        assert config.hidden_dim % config.num_heads==0
        assert config.num_heads % config.num_key_values==0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_key_values = config.num_key_values 
        self.head_dim = self.hidden_dim//self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_layers
        
        # modules
        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config,layer_idx = i) for i in range(self.num_layers)])
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.layer_norm = LlamaRMSNorm(self.hidden_dim)
        
        # initializations
        self.init_rope(config.rope_theta)
        
        # check env variables
        if os.getenv('MERGED_QKV_WEIGHT', '1') == '1':
            print("Initializing with merged qkv weights")
        else:
            print("Initializing with seperate qkv weights")
    
    def init_rope(self,rope_theta = 500000.0):
        cos, sin = get_cos_sin(self.max_position_embeddings, self.head_dim, base = rope_theta) # [max_position_embeddings, head_dim]
        cos = cos.to(device)
        sin = sin.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, input_ids, attention_mask, debug_arguments):
        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)

        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        variables = debug_arguments['variables']
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
            x = layer(x, cos, sin, attention_mask, debug_arguments)  # [batch_size, seq_length, hidden_dim]

        x = self.layer_norm(x)
        # print("After Final Layer Norm:", x)

        logits = self.lm_head(x)
        # print("Output Logits:", logits)

        return logits  # [batch_size, seq_length, vocab_size]