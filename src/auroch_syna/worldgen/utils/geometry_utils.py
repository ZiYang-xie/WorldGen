"""3D geometry utilities: ray generation, vector ops, depth alignment, mesh construction."""
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d


def pano_unit_rays(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Generate unit direction vectors for every pixel of an equirectangular image.

    Returns (h, w, 3) tensor in (x, y, z) right-handed convention.
    """
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    phi   = uu * 2 * torch.pi - torch.pi
    theta = vv * torch.pi - torch.pi / 2

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)  # (h, w, 3)


def batch_nearest_dot(src_dirs: torch.Tensor, query_dirs: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    """For each query direction, return the index of the closest source direction (max dot product).

    Processes queries in batches to avoid OOM on large tensors.
    """
    src_dirs   = F.normalize(src_dirs, dim=1)
    query_dirs = F.normalize(query_dirs, dim=1)
    idx = []
    for start in range(0, query_dirs.shape[0], batch):
        q = query_dirs[start:start + batch]
        sim = torch.mm(q, src_dirs.t())
        idx.append(sim.argmax(dim=1))
    return torch.cat(idx, dim=0)


def depth_match(init_pred, bg_pred, mask: np.ndarray):
    """Scale bg_pred.distance so its low-percentile depth aligns with init_pred in the masked region."""
    valid_mask    = (mask > 0)
    init_distance = init_pred.distance[valid_mask]
    bg_distance   = bg_pred.distance[valid_mask]

    init_mask = init_distance < torch.quantile(init_distance, 0.3)
    bg_mask   = bg_distance   < torch.quantile(bg_distance,   0.3)
    scale = init_distance[init_mask].median() / bg_distance[bg_mask].median()
    bg_pred.distance = bg_pred.distance * scale
    return bg_pred


def convert_rgbd2mesh_panorama(
    rgb: torch.Tensor,
    distance: torch.Tensor,
    rays: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_size: int = 4096,
    device: Literal["cuda", "cpu"] = "cuda",
) -> o3d.geometry.TriangleMesh:
    """Convert panoramic RGB-D data (image, distance, rays) into an Open3D triangle mesh.

    Args:
        rgb:      (H, W, 3) float tensor, values in [0, 1].
        distance: (H, W) distance map in metres.
        rays:     (H, W, 3) unit ray directions from the origin.
        mask:     (H, W) optional boolean tensor; True marks regions to exclude.
        max_size: Inputs are downscaled so max(H, W) <= max_size.
        device:   'cuda' or 'cpu'.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    if distance.ndim != 2:
        raise ValueError("distance must be HxW")
    if rays.ndim != 3 or rays.shape[2] != 3:
        raise ValueError("rays must be HxWx3")
    if rgb.shape[:2] != distance.shape[:2] or rgb.shape[:2] != rays.shape[:2]:
        raise ValueError("rgb, distance, and rays must share the same H and W")
    if mask is not None:
        if mask.ndim != 2 or mask.shape[:2] != rgb.shape[:2]:
            raise ValueError("mask shape must match rgb")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor")

    rgb      = rgb.to(device)
    distance = distance.to(device)
    rays     = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W  = distance.shape
    scale = max_size / max(H, W) if max(H, W) > max_size else 1.0

    def _interp(x, sf):
        return F.interpolate(x, scale_factor=sf, mode="bilinear",
                             align_corners=False, recompute_scale_factor=False)

    rgb_r  = _interp(rgb.permute(2, 0, 1).unsqueeze(0), scale).squeeze(0).permute(1, 2, 0)
    dist_r = _interp(distance.unsqueeze(0).unsqueeze(0), scale).squeeze(0).squeeze(0)
    rays_r = _interp(rays.permute(2, 0, 1).unsqueeze(0), scale).squeeze(0).permute(1, 2, 0)
    rays_r = rays_r / (torch.linalg.norm(rays_r, dim=-1, keepdim=True) + 1e-8)

    if mask is not None:
        mask_r = (_interp(mask.unsqueeze(0).unsqueeze(0).float(), scale).squeeze() > 0.5)
    else:
        mask_r = None

    H_new, W_new  = dist_r.shape
    vertices      = dist_r.reshape(-1, 1) * rays_r.reshape(-1, 3)
    vertex_colors = rgb_r.reshape(-1, 3)

    row = torch.arange(0, H_new - 1, device=device).repeat(W_new - 1)
    col = torch.arange(0, W_new - 1, device=device).repeat_interleave(H_new - 1)
    tl, tr, bl, br = row * W_new + col, row * W_new + col + 1, (row + 1) * W_new + col, (row + 1) * W_new + col + 1

    if mask_r is not None:
        keep = ~(mask_r[row, col] | mask_r[row, col + 1] |
                 mask_r[row + 1, col] | mask_r[row + 1, col + 1])
        idx = keep.nonzero(as_tuple=False).squeeze(-1)
        tl, tr, bl, br = tl[idx], tr[idx], bl[idx], br[idx]

    faces = torch.cat([torch.stack([tl, tr, bl], dim=1),
                       torch.stack([tr, br, bl], dim=1)], dim=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh.triangles     = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh
