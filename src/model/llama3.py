import os
from src.generation.decode import KVcache
from src.nn.layer_norm import LlamaRMSNorm, TritonRMSNorm
from src.parallel.tensor_parallel.initialize import get_model_parallel_world_size
from src.parallel.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
import torch 
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb
import lovely_tensors as lt
from lovely_tensors import set_config
set_config(precision=6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32
init_method = init.xavier_normal_

def apply_rotary_pos_emb(x, cos, sin):
    batch_size, num_head, seq_length, head_dim = x.size()
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

def flash_attention(q, k, v, causal = True):
    q = q.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    k = k.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    v = v.permute(0, 2, 1, 3) # [batch_size, seq_length, num_head , head_dim]
    return flash_attn_func(q, k, v, causal=causal)

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
        model_parallel_size = get_model_parallel_world_size()
        self.num_local_heads = config.num_heads // model_parallel_size # TP parallelism
        self.num_local_kv_heads = config.num_key_values // model_parallel_size # TP parallelism
        self.is_merged_qkv_weight = os.getenv('MERGED_QKV_WEIGHT', '1')
        if self.is_merged_qkv_weight  == '1': 
            self.qkv_proj = nn.Linear(config.hidden_dim, self.num_heads*self.head_dim + 2*self.num_key_values*self.head_dim, bias=False)
        else:
            # self.q_proj = nn.Linear(config.hidden_dim, self.num_heads*self.head_dim, bias=False)
            # self.k_proj = nn.Linear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False)
            # self.v_proj = nn.Linear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False)
            self.q_proj = ColumnParallelLinear(config.hidden_dim, self.num_heads*self.head_dim, bias=False, gather_output=False, init_method=init_method) # why the init method is x? Xavier is better?
            self.k_proj = ColumnParallelLinear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False, gather_output=False, init_method=init_method)
            self.v_proj = ColumnParallelLinear(config.hidden_dim, self.num_key_values*self.head_dim, bias=False, gather_output=False, init_method=init_method)
        # if os.getenv('FLASH_ROPE', '1') == '1':
        #     self.flash_rope = FlashRotaryEmbedding(dim=self.head_dim, interleaved=False, base=500000.0)
        
        # self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_dim, bias=False, input_is_parallel=True, init_method=init_method)
        self.kv_cache = None
        self.layer_idx = layer_idx
        
        ## TODO support mask
    
    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        if self.is_merged_qkv_weight == '1':
            qkv = self.qkv_proj(x) # [batch_size, seq_length, num_heads*head_dim + 2*num_key_values*head_dim]
            q, k, v = torch.split(qkv,
                [
                    self.num_local_heads * self.head_dim,
                    self.num_local_kv_heads * self.head_dim,
                    self.num_local_kv_heads * self.head_dim,
                ],
                dim=-1,
            ) # [batch_size, seq_length, num_heads*head_dim] / [batch_size, seq_length, num_key_values*head_dim] / [batch_size, seq_length, num_key_values*head_dim]
        else:
            q = self.q_proj(x) # [batch_size, seq_length, num_heads*head_dim]
            k = self.k_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
            v = self.v_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        if os.getenv('FLASH_ROPE', '1') != '1':
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim).transpose(1, 2)       # [batch_size, num_heads, seq_length, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_key_values, seq_length, head_dim]
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        else:
            q = q.view(batch_size, seq_length, self.num_local_heads, self.head_dim)       # [batch_size, seq_length, num_heads, head_dim]
            k = k.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim)  # [batch_size, seq_length, num_key_values, head_dim]
            q = apply_rotary_emb(q,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_heads, head_dim]
            k = apply_rotary_emb(k,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_key_values, head_dim]
            q = q.transpose(1, 2)                                                                   # [batch_size, num_heads, seq_length, head_dim]
            k = k.transpose(1, 2)                                                                   # [batch_size, num_key_values, seq_length, head_dim]
            v = v.view(batch_size, seq_length, self.num_local_kv_heads, self.head_dim).transpose(1,2)   # [batch_size, num_key_values, seq_length, head_dim]
        if self.kv_cache is not None:
            # update kv_cache, and get stored k, v
            assert position_ids is not None, "position_ids should be provided to update kv_cache"
            k, v = self.kv_cache.update_cache_get_kv(k, v, position_ids)
        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=1) # [batch_size, num_heads, seq_length, head_dim]
        if os.getenv('ATTENTION', 'SDPA') == 'SDPA':
            causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal) # [batch_size, num_heads, seq_length, head_dim]
            out = out.transpose(1, 2) # [batch_size, seq_length, num_heads, head_dim]
        else:
            causal = True if q.size(2) == k.size(2) else False # During decoding phase. The lenghth of q is usually 1. 
            out = flash_attention(q, k, v, causal = causal) # [batch_size, seq_length, num_heads, head_dim] 
        out = out.reshape(batch_size, seq_length, self.num_local_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        return out


class LLaMAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.merged_gate_up = os.getenv('MERGED_GATE_UP_WEIGHT', '1') == '1'
        if self.merged_gate_up:
            self.gate_up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim*2, bias=False)
        else:
            # self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            # self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
            self.up_proj = ColumnParallelLinear(config.hidden_dim, config.intermediate_dim, bias=False, gather_output=False, init_method=init_method)
            self.gate_proj = ColumnParallelLinear(config.hidden_dim, config.intermediate_dim, bias=False, gather_output=False, init_method=init_method)
        # self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_dim, config.hidden_dim, bias=False, input_is_parallel=True, init_method=init_method)
        
    def forward(self, x):
        if  self.merged_gate_up:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    # RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
    def __init__(self, config, layer_idx):
        super().__init__()
        if os.getenv('TRITONRMSNORM', '1') == '1':
            RMSNorm = TritonRMSNorm
        else:
            RMSNorm = LlamaRMSNorm
        self.input_layernorm = RMSNorm(config.hidden_dim)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim)
        self.attention = CausalSelfAttention(config, layer_idx = layer_idx)
        self.mlp = LLaMAMLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin, attention_mask = None, position_ids = None):
        x = x + self.attention(self.input_layernorm(x), cos, sin, attention_mask=attention_mask, position_ids=position_ids) # Attention 
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
        self.model_config = config
        
        # modules
        # self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.word_embedding = VocabParallelEmbedding(self.vocab_size, self.hidden_dim, init_method=init_method)
        self.layers = nn.ModuleList([DecoderLayer(config,layer_idx = i) for i in range(self.num_layers)])
        # self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.lm_head = ColumnParallelLinear(self.hidden_dim, self.vocab_size, bias=False, gather_output=True, init_method=init_method) # we can also not gather the output. TODO: add vocab_parallel_cross_entropy
        self.layer_norm = TritonRMSNorm(self.hidden_dim) if os.getenv('TRITONRMSNORM', '1') == '1' else LlamaRMSNorm(self.hidden_dim)
        
        # initializations
        self.init_rope(config.rope_theta)
    
    def init_rope(self,rope_theta = 500000.0):
        self.cos, self.sin = get_cos_sin(self.max_position_embeddings, self.head_dim, base = rope_theta) # [max_position_embeddings, head_dim]
        
    def initialize_kv_cache(self, batch_size, seq_len):
        for i in range(self.num_layers):
            self.layers[i].attention.kv_cache = KVcache(batch_size, seq_len, num_key_values=self.num_key_values, head_dim=self.head_dim, device=device, dtype=dtype)

    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        batch_size, seq_length = input_ids.size()
        x = self.word_embedding(input_ids)
        if position_ids is not None:
            # Pass specific position_ids to the model. Especially useful for generation during decoding phase. 
            # Like choose the correct RoPE for the current position, update kv_cache
            cos, sin = self.cos[position_ids], self.sin[position_ids]
        else:
            # Use the default position ids
            cos, sin = self.cos[:seq_length], self.sin[:seq_length]
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask, position_ids)  # [batch_size, seq_length, hidden_dim]
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits  # [batch_size, seq_length, vocab_size]     

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, use_cache=True):
        if use_cache == False:
            for i in range(max_new_tokens):
                next_token_id, next_token_logits = self.generate_one_token(input_ids)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            return input_ids
        else:
            self.initialize_kv_cache(input_ids.size(0), self.max_position_embeddings)
            
            # Prefilling phase
            position_ids = torch.arange(input_ids.size(1)).to(device)
            next_token_id, next_token_logits = self.generate_one_token(input_ids, position_ids=position_ids)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            
            # Decoding phase
            for i in range(max_new_tokens-1):
                position_ids = torch.tensor([input_ids.size(1)-1]).to(device)
                next_token_id, next_token_logits = self.generate_one_token(input_ids[0][-1].view(1,-1), position_ids)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

            return input_ids

    @torch.inference_mode()
    def generate_one_token(self, input_ids, position_ids=None):
        output_logits = self(input_ids, position_ids = position_ids)
        next_token_id = torch.argmax(output_logits[:, -1, :], dim=-1)
        return next_token_id, output_logits
