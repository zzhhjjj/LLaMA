# %%
import os
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
import lovely_tensors as lt
from lovely_tensors import set_config
set_config(precision=6)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    i = torch.arange(head_dim//2,dtype=torch.int64).float().unsqueeze(0).to(device) # [1, head_dim//2]
    theta = 1.0/(base**(2*i/head_dim)) # [1, head_dim//2]
    position = torch.arange(seq_length).unsqueeze(1).to(device) # [seq_length, 1]
    return torch.cos(position*theta).repeat(1,2), torch.sin(position*theta).repeat(1,2) # [seq_length, head_dim], [seq_length, head_dim]

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
        self.is_merged_qkv_weight = os.getenv('is_merged_qkv_weight', '1')
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
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        # k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        # v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        k = repeat_kv(k,self.num_heads // self.num_key_values)
        v = repeat_kv(v,self.num_heads // self.num_key_values)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, num_heads, seq_length, head_dim]
        # out = flash_attention(q, k, v) # [batch_size, num_heads, seq_length, head_dim] 
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.num_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
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
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        
    def forward(self, x):
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
        x = x + self.attention(self.input_layernorm(x),cos,sin, debug_arguments = debug_arguments) # Attention 
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x
    
class LLaMA(nn.Module):
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
    
    def init_rope(self,rope_theta = 500000.0):
        self.cos, self.sin = get_cos_sin(self.max_position_embeddings, self.head_dim, base = rope_theta) # [max_position_embeddings, head_dim]
        
    def forward(self, input_ids, attention_mask, debug_arguments=None):
        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)
        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)  # [batch_size, seq_length, hidden_dim]
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits  # [batch_size, seq_length, vocab_size]
# %%

