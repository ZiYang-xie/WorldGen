from PIL import Image
import cv2
import torch
import numpy as np
from worldgen.utils import pano_to_cube, cube_to_pano

def build_inpaint_model(device: torch.device = 'cuda'):
    from worldgen.models.inpaint_model import LaMa
    model = LaMa(device=device)
    return model

def inpaint_image(model, image: Image.Image, mask: Image.Image) -> Image.Image:
    original_size = image.size
    inpainted_image = model.infer(np.array(image), np.array(mask))
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image = inpainted_image.resize(original_size)
    return inpainted_image

@torch.inference_mode()
def inpaint_pano(model, image: Image.Image, mask: np.ndarray):
    H, W = image.height, image.width
    assert (H / W == 0.5),  "Input image aspect ratio is not 2:1. Is it a panorama?"
    cube_face_res = H // 2
    mask = Image.fromarray(mask * 255).convert("L")

    print(f"Processing as panorama. Converting to cubemap (calculated face res: {cube_face_res}px)...")
    cube_faces = pano_to_cube(image, face_w=cube_face_res)
    cube_masks = pano_to_cube(mask, face_w=cube_face_res, mode='nearest')

    cube_inpainted_faces = []
    for face, mask in zip(cube_faces, cube_masks):
        inpainted_face = inpaint_image(model, face, mask)
        cube_inpainted_faces.append(inpainted_face)

    pano_inpainted_image = cube_to_pano(cube_inpainted_faces, h=H, w=W, mode='bilinear')
    pano_inpainted_image.save("pano_inpainted_image.png")
    return pano_inpainted_image

