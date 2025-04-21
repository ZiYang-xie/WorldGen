import py360convert
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure, draw

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