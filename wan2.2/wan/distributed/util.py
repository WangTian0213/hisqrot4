# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def _all_to_all_impl(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


class _AllToAllAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim, group, kwargs):
        ctx.scatter_dim = int(scatter_dim)
        ctx.gather_dim = int(gather_dim)
        ctx.group = group
        ctx.kwargs = dict(kwargs)
        return _all_to_all_impl(
            x,
            scatter_dim=ctx.scatter_dim,
            gather_dim=ctx.gather_dim,
            group=ctx.group,
            **ctx.kwargs,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all_impl(
            grad_output,
            scatter_dim=ctx.gather_dim,
            gather_dim=ctx.scatter_dim,
            group=ctx.group,
            **ctx.kwargs,
        )
        return grad_input, None, None, None, None


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    world_size = get_world_size()
    if world_size <= 1:
        return x
    return _AllToAllAutograd.apply(x, scatter_dim, gather_dim, group, kwargs)


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


class _GatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = int(dim)
        ctx.rank = get_rank()
        ctx.world_size = get_world_size()
        output = all_gather(input)
        return torch.cat(output, dim=ctx.dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size <= 1:
            return grad_output, None
        chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return chunks[ctx.rank].contiguous(), None


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    return _GatherForwardSplitBackward.apply(input, dim)
