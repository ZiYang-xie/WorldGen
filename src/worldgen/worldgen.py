import torch 
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from .pano_depth import build_depth_model, pred_pano_depth, pred_depth
from .pano_seg import build_segment_model, seg_pano_fg
from .pano_gen import build_pano_gen_model, gen_pano_image, build_pano_fill_model, gen_pano_fill_image
from .pano_sharp import build_sharp_model, predict_equirectangular
from .pano_inpaint import build_inpaint_model, inpaint_image
from .utils.splat_utils import convert_rgbd_to_gs, SplatFile, mask_splat, merge_splats
from .utils.general_utils import map_image_to_pano, resize_img, depth_match, convert_rgbd2mesh_panorama
from typing import Optional, Union

class WorldGen:
    def __init__(self, 
            mode: str = 't2s',
            use_sharp: bool = False,
            inpaint_bg: bool = False,
            lora_path: str = None,
            resolution: int = 1600,
            device: torch.device = 'cuda',
            low_vram: Optional[bool] = None,
        ):
        self.device = device
        self.depth_model = build_depth_model(device)
        self.mode = mode
        self.resolution = resolution

        # Set low_vram based on available VRAM if not specified
        if low_vram is None:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            low_vram = total_vram < 24
            print(f"Detected {total_vram:.1f}GB VRAM, {'enabling' if low_vram else 'disabling'} low VRAM mode")
        self.low_vram = low_vram

        if mode == 't2s':
            self.pano_gen_model = build_pano_gen_model(lora_path=lora_path, device=device, low_vram=low_vram)
        elif mode == 'i2s':
            self.pano_gen_model = build_pano_fill_model(lora_path=lora_path, device=device, low_vram=low_vram)
        else:
            raise ValueError(f"Invalid mode: {mode}, mode must be 't2s' or 'i2s'")

        self.use_sharp = use_sharp
        if use_sharp:
            self.sharp_model = build_sharp_model(device=device)

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
    
    def depth2mesh(self, predictions) -> o3d.geometry.TriangleMesh:
        rgb = predictions["rgb"] / 255.0
        distance = predictions["distance"]
        rays = predictions["rays"]
        mesh = convert_rgbd2mesh_panorama(rgb, distance, rays)
        return mesh
    
    def inpaint_bg_splat(self, pano_image: Image.Image, init_splat: SplatFile, init_pred: dict) -> SplatFile:
        fg_mask = seg_pano_fg(self.seg_processor, self.seg_model, pano_image, init_pred["distance"])
        edge_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=1) - cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=1)
        init_splat = mask_splat(init_splat, (1-edge_mask))
        
        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5,5), np.uint8), iterations=10)
        pano_bg = inpaint_image(self.inpaint_pipe, pano_image, dilated_fg_mask)
        bg_pred = pred_pano_depth(self.depth_model, pano_bg)
        bg_pred = depth_match(init_pred, bg_pred, (1-dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = mask_splat(pano_bg_splat, dilated_fg_mask)
        merged_splat = merge_splats(init_splat, occ_bg_splat)
        return merged_splat
    
    def _generate_world(self, pano_image: Image.Image, return_mesh: bool = False) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        if self.use_sharp:
            splat = predict_equirectangular(self.sharp_model, pano_image, device=self.device)
            return splat

        init_pred = pred_pano_depth(self.depth_model, pano_image)
        if return_mesh:
            mesh = self.depth2mesh(init_pred)
            return mesh

        splat = self.depth2gs(init_pred)
        if self.inpaint_bg:
            splat = self.inpaint_bg_splat(pano_image, splat, init_pred)
        return splat

    
    def generate_pano(self, prompt: str = "", image: Optional[Image.Image] = None) -> Image.Image:
        if self.mode == 't2s':
            assert image is None, "image is not supported for text-to-scene generation"
            pano_image = gen_pano_image(self.pano_gen_model, prompt=prompt, height=self.resolution//2, width=self.resolution)
        elif self.mode == 'i2s':
            assert image is not None, "image is required for image-to-scene generation"
            image = resize_img(image) # Limit the longest edge to 1024 to avoid OOM
            predictions = pred_depth(self.depth_model, image)
            pano_cond_img, cond_mask = map_image_to_pano(predictions, device=self.device)
            pano_image = gen_pano_fill_image(
                self.pano_gen_model, 
                image=pano_cond_img, 
                mask=cond_mask,
                prompt=prompt, 
                height=self.resolution//2, 
                width=self.resolution
            )

            # Remap original image to pano image with higher resolution
            map_height, map_width = pano_cond_img.height, pano_cond_img.width
            pano_image = pano_image.resize((map_width, map_height))
            pano_cond_img, mask = np.array(pano_cond_img), np.array(cond_mask) / 255.0
            pano_image = np.array(pano_image) * mask[:,:,None] + pano_cond_img * (1-mask[:,:,None])
            pano_image = Image.fromarray(pano_image.astype(np.uint8))
        else:
            raise ValueError(f"Invalid mode: {self.mode}, mode must be 't2s' or 'i2s'")
        return pano_image
    
    @torch.inference_mode()
    def generate_world(
        self, 
        prompt: str = "", 
        image: Optional[Image.Image] = None, 
        return_mesh: bool = False
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        pano_image = self.generate_pano(prompt, image)
        scene = self._generate_world(pano_image, return_mesh)
        return scene