# %%
import os
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

import lovely_tensors as lt
from lovely_tensors import set_config
set_config(precision=6)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_rotary_pos_emb(x, cos, sin):
    # take x of shape (batch_size, num_heads, seq_length, head_dim)
    # cos, sin of shape (seq_length, head_dim)
    batch_size, num_head, seq_length, head_dim = x.size()
    assert cos.size(0)==seq_length
    assert cos.size(1)==head_dim
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x = x * cos + rotate_half * sin
    return x

def get_cos_sin(seq_length, head_dim, base=500000.0):
    # take sequence length and head dimension as input.
    # return (seq_length, head_dim)
    assert head_dim%2==0
    i = torch.arange(head_dim//2).unsqueeze(0).to(device)
    position = torch.arange(seq_length).unsqueeze(1).to(device)
    angle = 1.0/(base**(2*i/head_dim))
    return torch.cos(position*angle).repeat(1,2), torch.sin(position*angle).repeat(1,2)

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
        self.qkv_proj = nn.Linear(config.hidden_dim, self.num_heads*self.head_dim + 2*self.num_key_values*self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layer_idx = layer_idx
        
        ## TODO support mask
    
    def forward(self, x, cos, sin, attention_mask=None, debug_arguments=None):
        if debug_arguments!=None:
            return self.debug_forward(x, cos, sin, attention_mask, debug_arguments)
        else:
            return self.normal_forward(x, cos, sin, attention_mask)
        
    def normal_forward(self, x, cos, sin, attention_mask=None):
        batch_size, seq_length, hidden_dim = x.size()
        qkv = self.qkv_proj(x) # [batch_size, seq_length, num_heads*head_dim + 2*num_key_values*head_dim]
        
        q, k, v = torch.split(qkv,
            [
                self.num_heads * self.head_dim,
                self.num_key_values * self.head_dim,
                self.num_key_values * self.head_dim,
            ],
            dim=-1,
        )
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_length, head_dim]
        k = k.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
        v = v.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.num_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        return out
    
    # debug_arguments = 
    # {
    #     'layers': [0,1,2,3, ... ], # layers to print
    #     'variables': [1,2,3, ...], # variables to print
    # }
    def debug_forward(self, x, cos, sin, attention_mask=None,debug_arguments=None):
        assert debug_arguments!=None and 'layers' in debug_arguments and 'variables' in debug_arguments
        layers = debug_arguments['layers']
        variables = debug_arguments['variables']
        
        batch_size, seq_length, hidden_dim = x.size()
        qkv = self.qkv_proj(x) # [batch_size, seq_length, num_heads*head_dim + 2*num_key_values*head_dim]
        
        q, k, v = torch.split(qkv,
            [
                self.num_heads * self.head_dim,
                self.num_key_values * self.head_dim,
                self.num_key_values * self.head_dim,
            ],
            dim=-1,
        )

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_length, head_dim]
        k = k.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
        v = v.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1, 2) # [batch_size, num_key_values, seq_length, head_dim]
        
        if self.layer_idx in layers and 2 in variables:
            print("Q reshaped:", q)
            print("K reshaped:", k)
            print("V reshaped:", v)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        if self.layer_idx in layers and 3 in variables:
            print("Q after rotary pos emb:", q)
            print("K after rotary pos emb:", k)

        k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1) # [batch_size, num_heads, seq_length, head_dim]

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [batch_size, num_heads, seq_length, head_dim]
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
        if debug_arguments!=None:
            return self.debug_forward(x, cos, sin, attention_mask, debug_arguments)
        x = x + self.attention(self.input_layernorm(x),cos,sin, debug_arguments = debug_arguments) # Attention 
        x = x + self.mlp(self.post_attention_layernorm(x)) # MLP
        return x

    def debug_forward(self, x, cos, sin, attention_mask = None, debug_arguments = None):
        layers = debug_arguments['layers']
        variables = debug_arguments['variables']
        if self.layer_idx in layers:
            print('Layer: ', self.layer_idx)
            print("Input hidden states: ", x)
            print("hidden states after LN:", self.input_layernorm(x))
            print("Attention output:", self.attention(self.input_layernorm(x),cos,sin))
        x = x + self.attention(self.input_layernorm(x),cos,sin, debug_arguments = debug_arguments)
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
        if debug_arguments:
            lt.monkey_patch()
            return self.debug_forward(input_ids, attention_mask, debug_arguments)
        else:
            return self.normal_forward(input_ids, attention_mask, debug_arguments)
    
    def normal_forward(self, input_ids, attention_mask, debug_arguments=None):
        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)
        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)  # [batch_size, seq_length, hidden_dim]
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits  # [batch_size, seq_length, vocab_size]

    def debug_forward(self, input_ids, attention_mask, debug_arguments=None):
        if debug_arguments is not None:
            lt.monkey_patch()

        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)

        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        print("cos:", cos)
        print("sin:", sin)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attention_mask, debug_arguments)  # [batch_size, seq_length, hidden_dim]

        x = self.layer_norm(x)
        print("After Final Layer Norm:", x)

        logits = self.lm_head(x)
        print("Logits:", logits)

        return logits  # [batch_size, seq_length, vocab_size]

    # def forward(self, input_ids, attention_mask, debug_arguments=None):
    #     batch_size, seq_length = input_ids.size()
    #     x = self.word_embedding(input_ids)
    #     cos, sin = self.cos[:seq_length], self.sin[:seq_length]
    #     for layer in self.layers:
    #         x = layer(x, cos, sin, attention_mask, debug_arguments) # [batch_size, seq_length, hidden_dim]
    #     x = self.layer_norm(x)
    #     logits = self.lm_head(x)
        
    #     return logits # [batch_size, seq_length, vocab_size]
    

# %%


