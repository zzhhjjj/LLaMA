import torch
import torch.distributed as dist
from torch import nn

class DataParallel(nn.Module):
    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.dp_world_size = dist.get_world_size(group=self.process_group)
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.process_group)
        grad /= self.dp_world_size
        return grad 