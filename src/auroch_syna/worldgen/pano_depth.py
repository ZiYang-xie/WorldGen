"""Depth estimation using DA-2 (SphereViT).

Exports:
    build_depth_model  — load and return the DA-2 model.
    pred_depth         — run depth estimation on any PIL image.
    DepthPrediction    — typed result dataclass returned by pred_depth.
"""
import torch
import numpy as np
from dataclasses import dataclass
from PIL import Image
from da2.model.spherevit import SphereViT
from .constants import MAX_DEPTH_DISTANCE
from .utils.geometry_utils import pano_unit_rays

# Default config matching DA-2's inference settings
DA2_CONFIG = {
    "inference": {
        "min_pixels": 580000,
        "max_pixels": 620000
    },
    "spherevit": {
        "vit_w_esphere": {
            "input_dims": [1024, 1024, 1024, 1024],
            "hidden_dim": 512,
            "num_heads": 8,
            "expansion": 4,
            "num_layers_head": [2, 2, 2],
            "dropout": 0.0,
            "layer_scale": 0.0001,
            "out_dim": 64,
            "kernel_size": 3,
            "num_prompt_blocks": 1,
            "use_norm": False
        },
        "sphere": {
            "width": 1092,
            "height": 546,
            "hfov": 6.2832,
            "vfov": 3.1416
        }
    }
}


@dataclass
class DepthPrediction:
    """Structured result from pred_depth."""
    rgb: torch.Tensor       # (H, W, 3) uint8 on model.device
    distance: torch.Tensor  # (H, W)  metres, float32
    rays: torch.Tensor      # (H, W, 3) unit vectors, float32


def build_depth_model(device: torch.device = torch.device("cuda")):
    model = SphereViT.from_pretrained("haodongli/DA-2", config=DA2_CONFIG)
    model.eval()
    model = model.to(device)
    return model


def pred_depth(model, image: Image.Image) -> DepthPrediction:
    """Run DA-2 depth estimation on *image* and return a DepthPrediction.

    Works for both equirectangular panoramas and regular perspective images.
    The distance map is rescaled so its maximum equals MAX_DEPTH_DISTANCE.
    """
    rgb_np = np.array(image)
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # C, H, W
    rgb = rgb.unsqueeze(0).to(next(model.parameters()).dtype).to(model.device)

    with torch.autocast(model.device.type), torch.no_grad():
        distance = model(rgb)  # (1, H, W)

    distance = distance.squeeze(0).float()  # (H, W)
    distance = distance / distance.max() * MAX_DEPTH_DISTANCE
    h, w = distance.shape
    rays = pano_unit_rays(h, w, model.device)  # (H, W, 3)
    rgb_out = torch.tensor(np.array(image.resize((w, h))), device=model.device)

    return DepthPrediction(rgb=rgb_out, distance=distance, rays=rays)


# Backward-compatible alias used by callers that previously called
# pred_pano_depth specifically for equirectangular images.
pred_pano_depth = pred_depth


if __name__ == "__main__":
    model = build_depth_model()
    image = Image.open("data/background/timeless_desert.png")
    predictions = pred_depth(model, image)
    print(predictions)
