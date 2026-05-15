from PIL import Image
import torch
import argparse
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from .utils.general_utils import pano_to_cube, cube_to_pano

def build_segment_model(device: torch.device = 'cuda'):
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to(device)
    model.eval()
    return processor, model

@torch.inference_mode()
def segment_image_oneformer(processor, model, image: Image.Image):
    """Segments instances in a single image using OneFormer and returns a binary mask."""
    original_size = image.size
    device = model.device

    image = image.convert("RGB")
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
    outputs = model(**inputs)
    predicted_seg_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[original_size[::-1]])[0]
    assert predicted_seg_map.max() <= 255, "Segmentation map only supports 255 unique values"
    predicted_seg_map = Image.fromarray(predicted_seg_map.cpu().numpy().astype(np.uint8))
    return predicted_seg_map


@torch.inference_mode()
def seg_pano(processor, model, image: Image.Image):
    H, W = image.height, image.width
    assert (H / W == 0.5),  "Input image aspect ratio is not 2:1. Is it a panorama?"
    cube_face_res = H // 2

    print(f"Processing as panorama. Converting to cubemap (calculated face res: {cube_face_res}px)...")
    cube_faces = pano_to_cube(image, face_w=cube_face_res)

    cube_masks = []
    for i, face in enumerate(cube_faces):
        seg_map = segment_image_oneformer(processor, model, face)
        cube_masks.append(seg_map)

    pano_mask = cube_to_pano(cube_masks, h=H, w=W, mode='nearest')
    return pano_mask

def seg_pano_fg(processor, model, image: Image.Image, depth: torch.Tensor):
    '''
    Segment the foreground of a panorama image using semantic segmentation # and depth map.
    '''
    background_labels = [0, 2, 3, 5, 8, 9, 13, 21, 23, 16, 46] # ADE20K background labels
    pano_semantic_mask = seg_pano(processor, model, image)
    pano_semantic_mask = torch.tensor(np.array(pano_semantic_mask)).to(depth.device)
    semantic_labels = torch.unique(pano_semantic_mask).tolist()
    instance_labels = []

    depth_min = torch.quantile(depth, 0.05)
    depth_max = torch.quantile(depth, 0.95)

    for label in semantic_labels:
        if label in background_labels:
            continue
        mask = (pano_semantic_mask == label)
        instance_depth = depth[mask]
        if instance_depth.min() < depth_min or instance_depth.max() > depth_max:
            continue
        instance_labels.append(label)

    instance_labels = torch.tensor(instance_labels, device=depth.device)
    fg_mask = torch.isin(pano_semantic_mask, instance_labels).to(torch.uint8)
    fg_mask = fg_mask.cpu().numpy()
    return fg_mask
    