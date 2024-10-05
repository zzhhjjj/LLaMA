import contextlib
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Variable
from src.parallel.data_parallel.bucket import BucketManager

class DataParallel(nn.Module):
    def __init__(self, module, process_group, bucket_cap_mb=25, grad_type = torch.float32):
        super().__init__()
        self.module = module
        self.process_group = process_group # process group for gradient synchronization. could be data parallel group and context parallel group
        self.dp_world_size = dist.get_world_size(group=self.process_group)
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), self.process_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self):
        """
        Register backward hook: add a post hook to gradient accumulation nodes to sync gradients and manully sync gradients.
        Accumulation function for the gradients need to be stored so they don't go out of scope.
        ref: 
        https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        https://github.com/NVIDIA/Megatron-LM/issues/690 
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the all-reduce hook for backprop.
        """
        def param_hook(*unused):
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for the gradient synchronization to finish. 
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    # Execute the callback only once.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        Wait for the gradient synchronization to finish, and copy gradient to params.grad
        This function is called before the optimizer step and after the backward pass.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager.
        Zero the gradients of the model. 
        """
        self.bucket_manager.reset() 