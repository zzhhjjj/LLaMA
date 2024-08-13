import torch
import torch.nn as nn


class KVcache(nn.Module):
    """ a class to store key and value cache for a single layer
    """
    def __init__(self, batch_size, seq_len, num_key_values, head_dim, device, dtype):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(batch_size, num_key_values, seq_len, head_dim, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(batch_size, num_key_values, seq_len, head_dim, device=device, dtype=dtype), persistent=False)
        self.seq_len = 0
        
    def update_cache_get_kv(self, k, v, input_pos):
        k = self.k_cache.index_copy_(2, input_pos, k)[:,:,:input_pos[-1]+1,:] # [batch_size, num_key_values, curr_seq_len, head_dim]
        v = self.v_cache.index_copy_(2, input_pos, v)[:,:,:input_pos[-1]+1,:] # [batch_size, num_key_values, curr_seq_len, head_dim]
        return k , v
    
    def get_seq_length(self):
        return self.seq_len

    

