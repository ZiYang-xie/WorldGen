from PIL import Image
import torch
import numpy as np
from auroch_syna.runtime import get_logger
from .utils.general_utils import pano_to_cube, cube_to_pano

log = get_logger(__name__)


def build_inpaint_model(device=None):
    from .models.inpaint_model import LaMa
    if device is None:
        from auroch_syna.runtime import resolve_device
        device = resolve_device().name
    return LaMa(device=device)


def inpaint_image(model, image: Image.Image, mask: Image.Image) -> Image.Image:
    original_size = image.size
    inpainted_image = model.infer(np.array(image), np.array(mask))
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image = inpainted_image.resize(original_size)
    return inpainted_image


@torch.inference_mode()
def inpaint_pano(model, image: Image.Image, mask: np.ndarray, *, save_path: str | None = None):
    H, W = image.height, image.width
    assert H * 2 == W, "Input image aspect ratio is not 2:1. Is it a panorama?"
    cube_face_res = H // 2
    mask = Image.fromarray(mask * 255).convert("L")

    log.info("Inpainting panorama as cubemap (face_res=%dpx)", cube_face_res)
    cube_faces = pano_to_cube(image, face_w=cube_face_res)
    cube_masks = pano_to_cube(mask, face_w=cube_face_res, mode='nearest')

    cube_inpainted_faces = []
    for face, mask in zip(cube_faces, cube_masks):
        inpainted_face = inpaint_image(model, face, mask)
        cube_inpainted_faces.append(inpainted_face)

    pano_inpainted_image = cube_to_pano(cube_inpainted_faces, h=H, w=W, mode='bilinear')
    if save_path is not None:
        pano_inpainted_image.save(save_path)
    return pano_inpainted_image

