import os
import numpy as np
import cv2
import torch

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

class LaMa:
    @staticmethod
    def download():
        from iopaint.helper import download_model
        download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5)

    def init_model(self, device, **kwargs):
        from iopaint.helper import load_jit_model
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    def __init__(self, device: torch.device = 'cuda'):
        self.device = device
        self.init_model(device)

    @staticmethod
    def is_downloaded() -> bool:
        from iopaint.helper import get_cache_path_by_url
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def infer(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        from iopaint.helper import norm_img
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        return cur_res