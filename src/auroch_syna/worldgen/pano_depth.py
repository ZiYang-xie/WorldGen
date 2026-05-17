import torch
import numpy as np
from PIL import Image
from da2.model.spherevit import SphereViT
from auroch_syna.runtime import get_logger
from auroch_syna.runtime.torch_compat import safe_autocast
from .utils.general_utils import pano_unit_rays

log = get_logger(__name__)

MAX_DISTANCE = 20.0  # Scale DA-2's scale-invariant output so max distance = 20 meters

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

def build_depth_model(device=None):
    if device is None:
        from auroch_syna.runtime import resolve_device
        device = resolve_device().name
    model = SphereViT.from_pretrained("haodongli/DA-2", config=DA2_CONFIG)
    model.eval()
    model = model.to(device)
    return model

def pred_pano_depth(model, image: Image.Image):
    rgb_np = np.array(image)
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # C, H, W, normalized to [0, 1]
    rgb = rgb.unsqueeze(0).to(next(model.parameters()).dtype).to(model.device)

    with safe_autocast(model.device.type), torch.no_grad():
        distance = model(rgb)  # (1, H, W)

    distance = distance.squeeze(0).float()  # (H, W)
    distance = distance / distance.max() * MAX_DISTANCE
    h, w = distance.shape
    rays = pano_unit_rays(h, w, model.device)  # (H, W, 3)

    rgb_out = torch.tensor(np.array(image.resize((w, h))), device=model.device)

    return {
        "rgb": rgb_out,
        "depth": distance,
        "distance": distance,
        "rays": rays,
    }


# Historical alias: pred_depth used to be a byte-identical copy.
pred_depth = pred_pano_depth
