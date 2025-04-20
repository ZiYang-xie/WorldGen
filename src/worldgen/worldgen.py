import torch 
import numpy as np
import cv2
from PIL import Image
from .pano_depth import build_depth_model, pred_pano_depth, pred_depth
from .pano_seg import build_segment_model, seg_pano_fg
from .pano_gen import build_pano_gen_model, gen_pano_image, build_pano_fill_model, gen_pano_fill_image
from .pano_inpaint import build_inpaint_model, inpaint_image
from .utils import convert_rgbd_to_gs, map_image_to_pano, resize_img, SplatFile

from typing import Optional

class WorldGen:
    def __init__(self, 
                mode: str = 't2s',
                inpaint_bg: bool = False,
                device: torch.device = 'cuda',
                resolution: int = 1440,
        ):
        self.device = device
        self.depth_model = build_depth_model(device)
        self.mode = mode
        self.resolution = resolution

        if mode == 't2s':
            self.pano_gen_model = build_pano_gen_model(device)
        elif mode == 'i2s':
            self.pano_gen_model = build_pano_fill_model(device)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode must be 't2s' or 'i2s'")

        self.inpaint_bg = inpaint_bg
        if inpaint_bg:
            self.seg_processor, self.seg_model = build_segment_model(device)
            self.inpaint_pipe = build_inpaint_model(device)

    def depth2gs(self, predictions) -> SplatFile:
        rgb = predictions["rgb"]
        distance = predictions["distance"]
        rays = predictions["rays"]
        splat = convert_rgbd_to_gs(rgb, distance, rays)
        return splat
    
    def mask_splat(self, splat: SplatFile, mask: np.ndarray) -> SplatFile:
        H, W = mask.shape
        valid_mask = mask>0
        centers = splat.centers
        covariances = splat.covariances
        rgbs = splat.rgbs
        opacity = splat.opacities
        scales = splat.scales
        rotations = splat.rotations

        centers = centers.reshape(H, W, 3)[valid_mask]
        covariances = covariances.reshape(H, W, 3, 3)[valid_mask]
        rgbs = rgbs.reshape(H, W, 3)[valid_mask]
        opacity = opacity.reshape(H, W, 1)[valid_mask]
        scales = scales.reshape(H, W, 3)[valid_mask]
        rotations = rotations.reshape(H, W, 4)[valid_mask]

        splat = {
            "centers": centers,
            "covariances": covariances,
            "rgbs": rgbs,
            "opacities": opacity,
            "scales": scales,
            "rotations": rotations
        }
        return SplatFile(**splat)
    
    def merge_splats(self, splat1: SplatFile, splat2: SplatFile) -> SplatFile:
        return SplatFile(
            centers=np.concatenate([splat1.centers, splat2.centers], axis=0),
            covariances=np.concatenate([splat1.covariances, splat2.covariances], axis=0),
            rgbs=np.concatenate([splat1.rgbs, splat2.rgbs], axis=0),
            opacities=np.concatenate([splat1.opacities, splat2.opacities], axis=0),
            scales=np.concatenate([splat1.scales, splat2.scales], axis=0),
            rotations=np.concatenate([splat1.rotations, splat2.rotations], axis=0)
        )
    
    def depth_match(self, init_pred: dict, bg_pred: dict, mask: np.ndarray) -> dict:
        valid_mask = (mask > 0)
        init_distance = init_pred["distance"][valid_mask]
        bg_distance = bg_pred["distance"][valid_mask]

        init_mask = init_distance < torch.quantile(init_distance, 0.3)
        bg_mask = bg_distance < torch.quantile(bg_distance, 0.3)
        scale = init_distance[init_mask].median() / bg_distance[bg_mask].median()
        bg_pred["distance"] *= scale
        return bg_pred

    def _generate_world(self, pano_image: Image.Image) -> SplatFile:
        init_pred = pred_pano_depth(self.depth_model, pano_image)
        init_splat = self.depth2gs(init_pred)
        if not self.inpaint_bg:
            return init_splat

        fg_mask = seg_pano_fg(self.seg_processor, self.seg_model, pano_image, init_pred["distance"])
        edge_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=1) - cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        init_splat = self.mask_splat(init_splat, (1-edge_mask))
        
        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=10)
        pano_bg = inpaint_image(self.inpaint_pipe, pano_image, dilated_fg_mask)
        bg_pred = pred_pano_depth(self.depth_model, pano_bg)
        bg_pred = self.depth_match(init_pred, bg_pred, (1-dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = self.mask_splat(pano_bg_splat, dilated_fg_mask)
        merged_splat = self.merge_splats(init_splat, occ_bg_splat)
        return merged_splat
    
    def generate_pano(self, prompt: str = "", image: Optional[Image.Image] = None) -> Image.Image:
        if self.mode == 't2s':
            assert image is None, "image is not supported for text-to-scene generation"
            pano_image = gen_pano_image(self.pano_gen_model, prompt=prompt, height=self.resolution//2, width=self.resolution)
        elif self.mode == 'i2s':
            assert image is not None, "image is required for image-to-scene generation"
            image = resize_img(image)
            predictions = pred_depth(self.depth_model, image)
            pano_cond_img, cond_mask = map_image_to_pano(
                predictions, 
                equi_h=self.resolution//2, 
                equi_w=self.resolution, 
                device=self.device
            )
            pano_image = gen_pano_fill_image(
                self.pano_gen_model, 
                image=pano_cond_img, 
                mask=cond_mask, 
                prompt=prompt, 
                height=self.resolution//2, 
                width=self.resolution
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}, mode must be 't2s' or 'i2s'")
        return pano_image
    
    def generate_world(self, prompt: str = "", image: Optional[Image.Image] = None) -> SplatFile:
        pano_image = self.generate_pano(prompt, image)
        splat = self._generate_world(pano_image)
        return splat