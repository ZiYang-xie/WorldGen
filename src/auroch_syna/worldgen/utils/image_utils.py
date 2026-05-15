"""Image and panorama utilities: cubemap conversion, resizing, and panorama remapping."""
import py360convert
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure, draw

from .geometry_utils import pano_unit_rays, batch_nearest_dot


def pano_to_cube(pano_img: Image.Image, face_w: int, mode: str = "bilinear") -> list[Image.Image]:
    """Convert a panoramic PIL Image to a list of 6 cubemap face PIL Images."""
    pano_np = np.array(pano_img)
    cube_faces = py360convert.e2c(pano_np, face_w=face_w, mode=mode, cube_format="list")
    return [Image.fromarray(face) for face in cube_faces]


def cube_to_pano(cube_faces: list[Image.Image], h: int, w: int, mode: str = "bilinear") -> Image.Image:
    """Convert a list of 6 cubemap face PIL Images back to a panoramic PIL Image."""
    cube_np_faces = []
    for face in cube_faces:
        face_np = np.array(face)
        if face_np.ndim == 3 and face_np.shape[2] == 1:
            face_np = face_np.squeeze(axis=2)
        cube_np_faces.append(face_np)
    pano_np = py360convert.c2e(cube_np_faces, h=h, w=w, mode=mode, cube_format="list")
    return Image.fromarray(pano_np.astype(np.uint8))


def resize_img(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize a PIL image so its longer side equals max_size, preserving aspect ratio."""
    W, H = img.size
    if H > W:
        return img.resize((max_size, H * max_size // W))
    return img.resize((W * max_size // H, max_size))


def resize_img_and_rays(
    img: torch.Tensor,
    rays: torch.Tensor,
    equi_H: int,
    equi_W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize image and rays to match the angular resolution of a target panorama size.

    Args:
        img:    (H, W, 3) float tensor.
        rays:   (H, W, 3) per-pixel ray directions.
        equi_H: target panorama height.
        equi_W: target panorama width.

    Returns:
        img_resized, rays_resized — both tensors with the new spatial dimensions.
    """
    H, W = img.shape[:2]

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

    img_resized = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0),
        size=(H_new, W_new), mode="bilinear", align_corners=False,
    ).squeeze(0).permute(1, 2, 0)

    rays_resized = F.interpolate(
        rays.permute(2, 0, 1).unsqueeze(0),
        size=(H_new, W_new), mode="bilinear", align_corners=False,
    ).squeeze(0).permute(1, 2, 0)
    rays_resized = F.normalize(rays_resized, dim=-1)

    return img_resized, rays_resized


def fill_mask_from_contour(mask: torch.Tensor) -> torch.Tensor:
    """Fill the interior of mask contours; returns a filled boolean tensor."""
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    contours = measure.find_contours(mask_np, fully_connected="high")
    filled = np.zeros_like(mask_np, dtype=np.uint8)
    for contour in contours:
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1], filled.shape)
        filled[rr, cc] = 1
    return torch.from_numpy(filled)


def map_image_to_pano(
    predictions,
    crop_center: bool = False,
    map_h: int = 1024,
    map_w: int = 2048,
    nn_batch: int = 8192,
    device: torch.device = torch.device("cuda"),
) -> tuple[Image.Image, Image.Image]:
    """Project a DepthPrediction's RGB into an equirectangular panorama.

    Returns:
        pano_img:        equirectangular PIL Image (map_h × map_w).
        invalid_mask_img: greyscale PIL Image marking unmapped regions.
    """
    rays_src = predictions.rays
    rgb_src  = predictions.rgb.float()

    rgb_src, rays_src = resize_img_and_rays(rgb_src, rays_src, map_h, map_w)
    img_flat  = rgb_src.reshape(-1, 3)
    rays_flat = rays_src.reshape(-1, 3)

    x, y, z = rays_flat[:, 0], rays_flat[:, 1], rays_flat[:, 2]
    phi   = torch.atan2(x, z)
    theta = torch.asin(torch.clamp(y, -1.0, 1.0))
    u     = (phi / torch.pi + 1) / 2
    v     = theta / torch.pi + 0.5
    u_pix = (u * (map_w - 1)).long()
    v_pix = (v * (map_h - 1)).long()

    pano     = torch.zeros((map_h, map_w, 3), dtype=rgb_src.dtype, device=rgb_src.device)
    hit_mask = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb_src.device)
    pano[v_pix, u_pix]     = img_flat
    hit_mask[v_pix, u_pix] = True

    rays_pano = pano_unit_rays(map_h, map_w, device)
    hole_mask = ~hit_mask

    if hole_mask.any():
        rays_hole = rays_pano[hole_mask]
        nn_idx    = batch_nearest_dot(rays_flat, rays_hole, nn_batch)
        colours   = img_flat[nn_idx]
        hole_idx  = hole_mask.nonzero(as_tuple=False)
        pano[hole_idx[:, 0], hole_idx[:, 1]] = colours

    if crop_center:
        coords      = torch.stack((v_pix, u_pix), dim=-1)
        top_left, bottom_right = coords[0], coords[-1]
        valid_mask  = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb_src.device)
        valid_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = True
    else:
        valid_mask = F.max_pool2d(hit_mask.unsqueeze(0).float(), kernel_size=3, stride=1, padding=1).squeeze(0)
        valid_mask = fill_mask_from_contour(valid_mask).to(device)

    pano = pano * valid_mask[..., None]

    pano_img         = Image.fromarray(pano.clamp(0, 255).cpu().numpy().astype(np.uint8))
    invalid_mask_img = Image.fromarray(((1 - valid_mask.float()).cpu().numpy() * 255).astype(np.uint8))

    return pano_img, invalid_mask_img
