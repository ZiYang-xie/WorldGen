import torch
import numpy as np
from PIL import Image
from unik3d.models import UniK3D
from unik3d.utils.camera import Spherical

def build_depth_model(device: torch.device = 'cuda'):
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
    model.eval()
    model = model.to(device)
    return model

def pred_pano_depth(model, image: Image.Image):
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # C, H, W
    rgb = rgb.to(model.device)
    H, W = rgb.shape[1:]

    camera = Spherical(params=torch.tensor([1.0, 1.0, 1.0, 1.0, W, H, np.pi, np.pi/2]))
    predictions = model.infer(rgb, camera)
    h, w = predictions["depth"].shape[2:]

    rgb = torch.tensor(np.array(image.resize((w, h))), device=model.device)
    depth = predictions["depth"].squeeze(0).squeeze(0)
    distance = predictions["distance"].squeeze(0).squeeze(0)
    rays = predictions["rays"].squeeze(0).permute(1, 2, 0)

    results = {
        "rgb": rgb, # (H, W, 3)
        "depth": depth, # (H, W)
        "distance": distance, # (H, W)
        "rays": rays # (H, W, 3)
    }

    return results

def pred_depth(model, image: Image.Image):
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # C, H, W
    rgb = rgb.to(model.device)
    H, W = rgb.shape[1:]

    predictions = model.infer(rgb)
    h, w = predictions["depth"].shape[2:]

    rgb = torch.tensor(np.array(image.resize((w, h))), device=model.device)
    depth = predictions["depth"].squeeze(0).squeeze(0)
    distance = predictions["distance"].squeeze(0).squeeze(0)
    rays = predictions["rays"].squeeze(0).permute(1, 2, 0)

    results = {
        "rgb": rgb, # (H, W, 3)
        "depth": depth, # (H, W)
        "distance": distance, # (H, W)
        "rays": rays # (H, W, 3)
    }

    return results
if __name__ == "__main__":
    model = build_depth_model()
    image = Image.open("data/background/timeless_desert.png")
    predictions = pred_pano_depth(model, image)
    print(predictions)