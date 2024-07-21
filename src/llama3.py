# %%
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
def rotate_half(x):
    # [x1,x2,x3,x4] -> [x3,x4,x1,x2]
    head_size = x.shape[-1]
    assert head_size%2==0
    x1 = x[..., : head_size // 2]  
    x2 = x[..., head_size // 2 :]  
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # take x of shape (batch_size, num_heads, seq_length, head_dim)
    # cos, sin of shape (seq_length, head_dim)
    batch_size, num_head, seq_length, head_dim = x.size()
    assert cos.size(0)==seq_length
    assert cos.size(1)==head_dim
    x = x * cos + rotate_half(x) * sin
    return x

def get_cos_sin(seq_length, head_dim, theta=3.14):
    # take sequence length and head dimension as input.
    # return (seq_length, head_dim)
    assert head_dim%2==0
    i = torch.arange(head_dim//2).unsqueeze(0).to(device)
    position = torch.arange(seq_length).unsqueeze(1).to(device)
    angle = 1.0/(theta**(2*i/head_dim))
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
    attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1) # (batch_size, num_head, seq_length, seq_length)
    return torch.matmul(attention_weights, v).transpose(1,2).reshape(batch_size, seq_length, num_head*head_dim).contiguous() # (batch_size, seq_length, head_dim*num_head)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim//self.num_heads
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim*3)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        ## TODO support GQA, mask
        # self.num_queries = config.num_queries
    
    def forward(self, x, cos, sin, attention_mask=None):
        batch_size, seq_length, hidden_dim = x.size()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        out = scaled_dot_product_attention(q,k,v)
        out = self.out_proj(out)
        return out
    
class LLaMAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim)
        
    def forward(self, x):
        return self.down_proj(F.selu(self.gate_proj(x))* self.up_proj(x))

class DecoderLayer(nn.Module):
    # LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = torch.nn.LayerNorm(config.hidden_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp = LLaMAMLP(config)
    def forward(self, x, cos, sin, attention_mask = None):
        x = x + self.attention(self.layer_norm1(x),cos,sin) # Attention 
        x = x + self.mlp(self.layer_norm2(x)) # MLP
        return x
    
class LLaMA(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check 
        # assert self.hidden_dim%self.num_queries==0
        # assert self.num_queries%self.num_key_values==0 
        
        # params
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_queries = config.num_heads # implement later
        # self.num_key_values = config.num_key_values # implement later
        self.head_dim = self.hidden_dim//self.num_queries
        self.seq_length = config.seq_length
        self.num_layers = config.num_layers
        
        # modules
        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_layers)])
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim)
        
        # initialization
        self.init_rope()
    
    def init_rope(self):
        self.cos, self.sin = get_cos_sin(self.seq_length, self.head_dim) # (seq_length, head_dim)
        
    def forward(self, input_ids, attention_mask):
        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)
        cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask) # (batch_size, seq_length, hidden_dim)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits # (batch_size, seq_length, vocab_size)
    
