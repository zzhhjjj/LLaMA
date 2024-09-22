from typing import List, Optional
import os
import torch
from datetime import timedelta
import torch.distributed as dist


# TP/CP/PP/DDP parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_RANKS = None
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP_RANKS = None

def initialize_torch_distributed():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def initialize_process_groups(
    model_parallel_size: int,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
    data_parallel_size: int = 1,
    *,
    model_parallel_backend: Optional[str] = None,
    pipeline_backend: Optional[str] = None,
    cp_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None,
    timeout: Optional[timedelta] = None,
) -> None:
    
    assert pipeline_parallel_size == 1, "pipeline parallel is not supported yet"
    assert context_parallel_size == 1, "context parallel is not supported yet"
    assert model_parallel_size > 0
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    assert world_size == model_parallel_size * pipeline_parallel_size * context_parallel_size * data_parallel_size, "world size must be equal to the product of all parallel sizes"
    rank = torch.distributed.get_rank()
    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size))
        print("> initializing context parallel with size {}".format(context_parallel_size))
        print("> initializing pipeline with size {}".format(pipeline_parallel_size))
        print("> initializing ddp with size {}".format(data_parallel_size))
        print("")
    
    # Set global variables
    global _DATA_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP_RANKS
    global _MODEL_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_RANKS

    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    assert _PIPELINE_PARALLEL_GROUP is None, "pipeline parallel group is already initialized"
    
    # The order of initialization is important!!
    # TP -> CP -> PP -> DDP
    process_groups = torch.LongTensor(range(world_size)).reshape(data_parallel_size, pipeline_parallel_size, context_parallel_size, model_parallel_size)
    
    # This will return the rank of different parallel groups
    # the order is DDP, PP, CP, TP
    found = torch.where(process_groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]
    
    # Set a default timeout if it's not provided
    if timeout is None:
        timeout = timedelta(minutes=10)  
        
    # Create process groups
    for i in range(pipeline_parallel_size):
        for j in range(context_parallel_size):
            for k in range(model_parallel_size):
                # This function requires that all processes in the main group enter this function, even if they are not going to be members of the group.
                # https://pytorch.org/docs/stable/distributed.html
                group = torch.distributed.new_group(process_groups[:, i, j, k].tolist(), backend=ddp_backend, timeout=timeout) # ranks (list[int]) – List of ranks of group members.
                if i == found[1] and j == found[2] and k == found[3]: # the processes share the same rank other groups belongs to the same process group.
                    _DATA_PARALLEL_GROUP = group
    
    for i in range(data_parallel_size):
        for j in range(pipeline_parallel_size):
            for k in range(context_parallel_size):
                group = torch.distributed.new_group(process_groups[i, j, k, :].tolist(), backend=model_parallel_backend, timeout=timeout)
                if i == found[0] and j == found[1] and k == found[2]:
                    _MODEL_PARALLEL_GROUP = group
                    
    for i in range(data_parallel_size):
        for j in range(context_parallel_size):
            for k in range(model_parallel_size):
                ranks = process_groups[i, :, j, k].tolist()
                group = torch.distributed.new_group(ranks, backend=pipeline_backend, timeout=timeout)
                if i == found[0] and j == found[2] and k == found[3]:
                    _PIPELINE_PARALLEL_GROUP = group
                    _PIPELINE_PARALLEL_RANKS = ranks
                    
    for i in range(data_parallel_size):
        for j in range(pipeline_parallel_size):
            for k in range(model_parallel_size):
                ranks = process_groups[i, j, :, k].tolist()
                group = torch.distributed.new_group(ranks, backend=cp_backend, timeout=timeout)
                if i == found[0] and j == found[1] and k == found[3]:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GROUP_RANKS = ranks
                    
# Helper functions to get the process group
def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None or _PIPELINE_PARALLEL_GROUP is None or _CONTEXT_PARALLEL_GROUP is None:
        return False
    return True


def get_context_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the context parallel group the caller rank belongs to."""
    assert (
        _CONTEXT_PARALLEL_GROUP is not None
    ), "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_ranks() -> List[int]:
    """Return context parallel ranks for the context parallel group."""
    assert _CONTEXT_PARALLEL_GROUP_RANKS is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP_RANKS


def get_context_parallel_world_size() -> int:
    """Return world size for the context parallel group."""
    return torch.distributed.get_world_size(group=get_context_parallel_group())


def get_context_parallel_rank() -> int:
    """Return my rank for the context parallel group."""
    return torch.distributed.get_rank(group=get_context_parallel_group())


def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_pipeline_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_GROUP is not None, "pipeline parallel group is not initialized"
    return _PIPELINE_PARALLEL_GROUP


def get_pipeline_parallel_ranks() -> List[int]:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_RANKS is not None, "pipeline parallel group is not initialized"
    return _PIPELINE_PARALLEL_RANKS

def get_pipeline_parallel_world_size() -> int:
    """Return world size for the pipeline parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_parallel_group())

def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel() -> None:
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

    global _PIPELINE_PARALLEL_GROUP
    _PIPELINE_PARALLEL_GROUP = None
    global _PIPELINE_PARALLEL_RANKS
    _PIPELINE_PARALLEL_RANKS = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GROUP_RANKS
    _CONTEXT_PARALLEL_GROUP_RANKS = None