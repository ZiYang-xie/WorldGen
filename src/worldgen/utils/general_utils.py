import py360convert
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure, draw
from typing import Optional, Literal
import open3d as o3d

def pano_to_cube(pano_img: Image.Image, face_w: int, mode: str = 'bilinear') -> list[Image.Image]:
    """Converts a panoramic PIL Image to a list of 6 cubemap face PIL Images."""
    pano_np = np.array(pano_img)
    cube_faces = py360convert.e2c(pano_np, face_w=face_w, mode='bilinear', cube_format='list')
    cube_pil_faces = [Image.fromarray(face) for face in cube_faces]
    return cube_pil_faces

def cube_to_pano(cube_faces: list[Image.Image], h: int, w: int, mode: str = 'bilinear') -> Image.Image:
    """Converts a list of 6 cubemap face PIL Images (masks) back to a panoramic PIL Image."""
    cube_np_faces = []
    for face in cube_faces:
        face_np = np.array(face)
        if face_np.ndim == 2:
             pass
        elif face_np.ndim == 3 and face_np.shape[2] == 1:
             face_np = face_np.squeeze(axis=2)
        cube_np_faces.append(face_np)

    pano_np = py360convert.c2e(cube_np_faces, h=h, w=w, mode=mode, cube_format='list')
    pano_pil = Image.fromarray(pano_np.astype(np.uint8))
    return pano_pil


def resize_img(img: Image.Image, max_size=1024):
    W, H = img.size
    if H > W:
        img = img.resize((max_size, H * max_size // W))
    else:
        img = img.resize((W * max_size // H, max_size))
    return img

def resize_img_and_rays(img, rays, equi_H, equi_W):
    """
    Resize image and rays to match the angular resolution of a target panorama size.
    
    Args:
        img: (H, W, 3) tensor image
        rays: (H, W, 3) tensor of per-pixel rays
        equi_H: target panorama height (e.g., 512)
        equi_W: target panorama width (e.g., 1024)
        
    Returns:
        img_resized: resized image
        rays_resized: resized rays
    """
    H, W = img.shape[:2]
    device = img.device

    # Estimate input angular coverage
    h_center = rays[H // 2]
    v_center = rays[:, W // 2]

    phi_l = torch.atan2(h_center[0, 0], h_center[0, 2])
    phi_r = torch.atan2(h_center[-1, 0], h_center[-1, 2])
    horizontal_fov = (phi_r - phi_l).abs()

    theta_t = torch.asin(torch.clamp(v_center[0, 1], -1.0, 1.0))
    theta_b = torch.asin(torch.clamp(v_center[-1, 1], -1.0, 1.0))
    vertical_fov = (theta_b - theta_t).abs()

    px_per_rad_h = equi_W / (2 * torch.pi)
    px_per_rad_v = equi_H / torch.pi

    W_new = int(horizontal_fov * px_per_rad_h)
    H_new = int(vertical_fov * px_per_rad_v)

    img_resized = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=(H_new, W_new), mode='bilinear', align_corners=False)
    img_resized = img_resized.squeeze(0).permute(1, 2, 0)

    rays_resized = F.interpolate(rays.permute(2, 0, 1).unsqueeze(0), size=(H_new, W_new), mode='bilinear', align_corners=False)
    rays_resized = F.normalize(rays_resized.squeeze(0).permute(1, 2, 0), dim=-1)

    return img_resized, rays_resized

def pano_unit_rays(h, w, device):
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H,W) order here

    phi   = uu * 2 * torch.pi - torch.pi
    theta = vv * torch.pi - torch.pi / 2

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)                    # (h,w,3)

def batch_nearest_dot(src_dirs, query_dirs, batch=8192):
    """
    For each query vector find the index of the source vector with the
    largest dot product.  Works on CUDA or CPU.
    """
    src_dirs = F.normalize(src_dirs, dim=1)
    query_dirs = F.normalize(query_dirs, dim=1)
    idx = []
    for start in range(0, query_dirs.shape[0], batch):
        q = query_dirs[start:start + batch]                      # (b,3)
        sim = torch.mm(q, src_dirs.t())                          # (b,N)
        idx.append(sim.argmax(dim=1))
    return torch.cat(idx, dim=0)    

def fill_mask_from_contour(mask):
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    contours = measure.find_contours(mask_np, fully_connected="high")
    filled = np.zeros_like(mask_np, dtype=np.uint8)
    for contour in contours:
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1], filled.shape)
        filled[rr, cc] = 1

    return torch.from_numpy(filled)

def map_image_to_pano(predictions: dict,
                    crop_center: bool = False,
                    map_h: int = 1024,
                    map_w: int = 2048,
                    nn_batch: int = 8192,
                    device: torch.device = 'cuda'):
    rays_src = predictions["rays"]          
    rgb_src  = predictions["rgb"].float()

    rgb_src, rays_src = resize_img_and_rays(rgb_src, rays_src, map_h, map_w)
    H, W = rgb_src.shape[:2]
    img_flat  = rgb_src.reshape(-1, 3)
    rays_flat = rays_src.reshape(-1, 3)

    x, y, z = rays_flat[:, 0], rays_flat[:, 1], rays_flat[:, 2]
    phi   = torch.atan2(x, z)
    theta = torch.asin(torch.clamp(y, -1.0, 1.0))
    u = (phi / torch.pi + 1) / 2
    v = (theta / torch.pi + 0.5)
    u_pix = (u * (map_w - 1)).long()
    v_pix = (v * (map_h - 1)).long()

    pano = torch.zeros((map_h, map_w, 3),dtype=rgb_src.dtype, device=rgb_src.device)
    pano[v_pix, u_pix] = img_flat
    hit_mask = torch.zeros((map_h, map_w),dtype=torch.bool, device=rgb_src.device)
    hit_mask[v_pix, u_pix] = True

    rays_pano = pano_unit_rays(map_h, map_w, device)
    hole_mask = ~hit_mask
    valid_mask = hit_mask

    # fill holes
    if hole_mask.any():
        rays_hole = rays_pano[hole_mask]                       # (Nh,3)
        nn_idx = batch_nearest_dot(rays_flat, rays_hole, nn_batch)  # (Nh,)
        colours = img_flat[nn_idx]
        hole_idx = hole_mask.nonzero(as_tuple=False)
        pano[hole_idx[:, 0], hole_idx[:, 1]] = colours
    
    # two methods to fill holes
    if crop_center:
        # directly crop the center valid region
        coords = torch.stack((v_pix, u_pix), dim=-1)
        top_left, bottom_right = coords[0], coords[-1]
        valid_mask = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb_src.device)
        valid_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = True
    else:
        # fill holes by max pool and contour finding
        valid_mask = F.max_pool2d(valid_mask.unsqueeze(0).float(), kernel_size=3, stride=1, padding=1).squeeze(0)
        valid_mask = fill_mask_from_contour(valid_mask).to(device)

    pano = pano * valid_mask[..., None]

    pano_img = Image.fromarray(pano.clamp(0, 255).cpu().numpy().astype(np.uint8))
    invalid_mask = 1 - valid_mask.float()
    invalid_mask_img = Image.fromarray((invalid_mask.cpu().numpy() * 255).astype(np.uint8))

    return pano_img, invalid_mask_img

def depth_match(init_pred: dict, bg_pred: dict, mask: np.ndarray) -> dict:
    valid_mask = (mask > 0)
    init_distance = init_pred["distance"][valid_mask]
    bg_distance = bg_pred["distance"][valid_mask]

    init_mask = init_distance < torch.quantile(init_distance, 0.3)
    bg_mask = bg_distance < torch.quantile(bg_distance, 0.3)
    scale = init_distance[init_mask].median() / bg_distance[bg_mask].median()
    bg_pred["distance"] *= scale
    return bg_pred

def convert_rgbd2mesh_panorama(
    rgb: torch.Tensor,                       # (H, W, 3) RGB image, values [0, 1]
    distance: torch.Tensor,                  # (H, W) Distance map
    rays: torch.Tensor,                      # (H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    max_size: int = 4096,                    # Max dimension for resizing
    device: Literal["cuda", "cpu"] = "cuda", # Computation device
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert rgb.shape[:2] == distance.shape[:2] == rays.shape[:2], "Input shapes must match"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_resized = F.interpolate(
        rgb_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).permute(1, 2, 0)

    distance_resized = F.interpolate(
        distance_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).squeeze(0)

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    
    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), # Needs float for interpolation
            scale_factor=scale,
            mode="bilinear", # Or 'nearest' if sharp boundaries are critical
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(0).squeeze(0)
        mask_resized = mask_resized > 0.5 # Convert back to boolean
    else:
        mask_resized = None

    H_new, W_new = distance_resized.shape # Get new dimensions

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1) # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)         # (H*W, 3)
    vertices = distance_flat * rays_flat            # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3) # (H*W, 3)

    # --- Generate Mesh Faces (Triangles from Quads) ---
    row_indices = torch.arange(0, H_new - 1, device=device)
    col_indices = torch.arange(0, W_new - 1, device=device)
    row = row_indices.repeat(W_new - 1)
    col = col_indices.repeat_interleave(H_new - 1)


    tl = row * W_new + col      # Top-left
    tr = tl + 1                 # Top-right
    bl = tl + W_new             # Bottom-left
    br = bl + 1                 # Bottom-right

    # Apply mask if provided
    if mask_resized is not None:
        mask_tl = mask_resized[row, col]
        mask_tr = mask_resized[row, col + 1]
        mask_bl = mask_resized[row + 1, col]
        mask_br = mask_resized[row + 1, col + 1]

        quad_keep_mask = ~(mask_tl | mask_tr | mask_bl | mask_br)

        keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze(-1)
        tl = tl[keep_indices]
        tr = tr[keep_indices]
        bl = bl[keep_indices]
        br = br[keep_indices]

    # --- Create Triangles ---
    tri1 = torch.stack([tl, tr, bl], dim=1)
    tri2 = torch.stack([tr, br, bl], dim=1)
    faces = torch.cat([tri1, tri2], dim=0) 

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.remove_degenerate_triangles()

    return mesh_o3d