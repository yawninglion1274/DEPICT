"""Native dense 3D TV-L1 port, preserving the verified skimage equations."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _median3(flow):
    padded = F.pad(flow, (1, 1, 1, 1, 1, 1), mode="replicate")
    return padded.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1).flatten(-3).median(-1).values


@torch.no_grad()
def tvl1_3d_batch(reference, moving, *, attachment=15.0, tightness=0.3,
                  num_warp=5, num_iter=10, tol=1e-4, prefilter=False, flow0=None):
    if reference.shape != moving.shape or reference.ndim != 4:
        raise ValueError("Expected matching (batch, z, y, x) inputs")
    if reference.dtype != torch.float32 or moving.dtype != torch.float32:
        raise ValueError("Use float32 inputs")
    if min(reference.shape[1:]) < 2:
        raise ValueError("Each spatial dimension must have at least two voxels")
    n, depth, height, width = reference.shape
    sizes = (depth, height, width)
    flow = (reference.new_zeros((n, 3, *sizes)) if flow0 is None else flow0.clone())
    previous = flow
    gradient = torch.zeros_like(flow)
    projection = reference.new_zeros((n, 3, 3, *sizes))
    coordinates = torch.meshgrid(*[torch.linspace(-1, 1, s, device=reference.device)
                                  for s in sizes], indexing="ij")
    base = torch.stack(coordinates[::-1], dim=-1)[None]
    scale = reference.new_tensor([2 / (s - 1) for s in sizes]).view(1, 3, 1, 1, 1)
    dt, f0, f1 = 1 / 6, attachment * tightness, (1 / 6) / tightness
    for _ in range(num_warp):
        if prefilter:
            flow = _median3(flow)
        grid = base + (flow * scale).flip(1).movedim(1, -1)
        warped = F.grid_sample(moving[:, None], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)[:, 0]
        del grid
        image_gradient = torch.stack(torch.gradient(warped, dim=(1, 2, 3)), dim=1)
        norm_image = image_gradient.square().sum(1)
        norm_image.masked_fill_(norm_image == 0, 1)
        rho0 = warped - reference - (image_gradient * flow).sum(1)
        for _ in range(num_iter):
            rho = rho0 + (image_gradient * flow).sum(1)
            coefficient = (rho / norm_image).clamp(-f0, f0)
            # skimage aliases the auxiliary field before copying for regularisation.
            auxiliary = flow
            auxiliary.sub_(coefficient[:, None] * image_gradient)
            flow = auxiliary.clone()
            for component in range(3):
                for _ in range(2):
                    for axis in range(3):
                        target = [slice(None)] * 4
                        target[axis + 1] = slice(0, -1)
                        gradient[:, axis][tuple(target)] = torch.diff(flow[:, component], dim=axis + 1)
                    norm = gradient.square().sum(1).sqrt().mul_(f1).add_(1)
                    p = projection[:, component]
                    p.sub_(dt * gradient).div_(norm[:, None])
                    divergence = -p.sum(1)
                    for axis in range(3):
                        target = [slice(None)] * 4
                        source = [slice(None)] * 4
                        target[axis + 1] = slice(1, None)
                        source[axis + 1] = slice(0, -1)
                        divergence[tuple(target)] += p[:, axis][tuple(source)]
                    flow[:, component] = auxiliary[:, component] + divergence
        previous.sub_(flow)
        error = previous.square().flatten(1).sum(1)
        if bool((error < tol * np.prod(sizes)).all()):
            break
        previous = flow
    return flow


@torch.no_grad()
def tvl1_3d_pyramid(reference, moving, **kwargs):
    """Coarse-to-fine version; skimage pyramids and nearest flow resizing on CPU.

    Pyramid preparation is inexpensive relative to optimisation; using skimage's
    exact pyramid and flow resizing preserves the established reference recipe.
    """
    from skimage.registration._optical_flow_utils import _get_pyramid, _resize_flow
    device = reference.device
    refs = _get_pyramid(reference[0].cpu().numpy())
    movs = _get_pyramid(moving[0].cpu().numpy())
    flow = None
    for ref, mov in zip(refs, movs):
        if flow is not None:
            flow = torch.from_numpy(_resize_flow(flow[0].cpu().numpy(), ref.shape))[None].to(device)
        flow = tvl1_3d_batch(torch.from_numpy(ref)[None].to(device),
                             torch.from_numpy(mov)[None].to(device), flow0=flow, **kwargs)
    return flow
