import contextlib
import torch
import torch.distributed as dist
from torch import nn

class DataParallel(nn.Module):
    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.process_group = process_group # process group for gradient synchronization. could be data parallel group and context parallel group
        self.dp_world_size = dist.get_world_size(group=self.process_group)
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        # no synchronization required(except for last epoch of gradient accumulation)
        # 324K tokens/s/gpu -> 334K tokens/s/gpu
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.process_group)
            grad /= self.dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True