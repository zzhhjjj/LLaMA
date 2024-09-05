import torch.distributed as dist
import torch
from .utils import get_model_parallel_group, get_tensor_parallel_group, split_tensor_along_last_dim

# Reference : Fairscale, Megatron-LM
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/mappings.py

def _reduce(input_):
    """All-reduce the input tensor across model parallel(TP) group."""

    tp_group = get_tensor_parallel_group()
    
    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(tp_group) == 1:
        return input_

    # All-reduce.
    dist.all_reduce(input_, group=tp_group)

    return input_

class _CopyToModelParallelRegion(torch.autograd.Function):
    # copy in forward pass, all reduce in backward pass
    @staticmethod
    def forward(ctx, input_):
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output

def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output

def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)



# Two functions in TP mode
def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


# Two functions in TP-SP mode
def scatter_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)
