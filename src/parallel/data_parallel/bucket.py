from typing import List
import torch
import torch.distributed as dist
from torch import nn

class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        self.params = set(params)    # set of parameters in the bucket
        self.params_with_grad_ready = set() # set of parameters that have gradients ready. launch all reduce when all parameters have gradients ready
        self.grad_data = grad_data  # the tensor to store the gradients
        self.process_group = process_group  # the process group for gradient synchronization
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.handle = None
        
        self.reset()
    
    # launch async allreduce operation to synchronize gradients
    def sync_gradient(self) -> None:
        assert self.handle is None
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)
        self.grad_data /= self.process_group_size
    
    def reset(self) -> None:
        self.handle = None
        self.params_with_grad_ready.clear()
        self.grad_data.zero_()
    
    # wait for the allreduce operation to finish    
    def wait(self) -> None:
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        self.handle.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        self.params = list(params) # generator to list
        self.buckets = []
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.params_to_bucket_location = {} # map parameter to (start, end, bucket_idx) 
        self.bucket_size = bucket_size
        self.bucket_sizes = None # the actual size of each bucket
        self.grad_data_list = [] # the tensors to store the gradients. each one is for one bucket
        self.grad_type = grad_type
        # divide the gradients into buckets
        self._initialize_buckets()
        
        assert type(self.params) == list # not generator
    
    # this method update slef.params_to_bucket_location. 
    def _initialize_buckets(self) -> None:
        cur_bucket_size = 0 
        cur_bucket_idx = 0
        
        # first, we want to know where each gradient goes
        for param in self.params:
            if not param.requires_grad:
                continue
            # when the current bucket is empty, add the parameter to the current bucket
            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
                continue
            
            # if the current parameter cannot fit in the current bucket, create a new bucket
            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
            else:
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                cur_bucket_size += param.numel()

        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)
        
        # create the tensor where we store the gradients, and the bucket on top of the tensor 
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device='cuda'))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))
        
        # create the view of the gradients for each parameter
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            # param.main_grad is used for gradient calculation. Because we use mixed precision training which is not supported by PyTorch by default
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)
            
    # get view of shape from a 1D tensor
    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        return tensor[start:end].view(shape)
    
    def reset(self) -> None:
        for bucket in self.buckets:
            bucket.reset()
    
    def wait(self) -> None:
        """
        wait for all the buckets to finish the allreduce operation
        """
        for bucket in self.buckets:
            bucket.wait()
    
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        bucket_idx = self.params_to_bucket_location[param][2]
        bucket = self.buckets[bucket_idx]
        bucket.mark_param_as_ready(param)


















