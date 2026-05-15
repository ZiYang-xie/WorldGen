"""High-level orchestrator that coordinates the full text/image → 3D scene pipeline.

Entry point: instantiate WorldGen, then call generate_world().
"""
import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from .device_utils import resolve_device
from .model_client import ModelClient
from .pano_depth import DepthPrediction
from .utils.splat_utils import convert_rgbd_to_gs, SplatFile, mask_splat, merge_splats
from .utils.image_utils import map_image_to_pano, resize_img
from .utils.geometry_utils import depth_match, convert_rgbd2mesh_panorama
from .model_handle import DepthHandle, PanoGenHandle, SegHandle, InpaintHandle, SharpHandle
from typing import Optional, Union


class WorldGen:
    def __init__(
        self,
        mode: str = "t2s",
        use_sharp: bool = False,
        inpaint_bg: bool = False,
        lora_path: str = None,
        resolution: int = 1600,
        device: torch.device | str | None = None,
        low_vram: Optional[bool] = None,
    ):
        self.device = resolve_device(device)

        if low_vram is None and self.device.type == "cuda":
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                low_vram = total_vram < 24
                print(f"Detected {total_vram:.1f}GB VRAM, {'enabling' if low_vram else 'disabling'} low VRAM mode")
            except (AttributeError, RuntimeError):
                low_vram = False
        self.low_vram = low_vram or False

        self.model_client = ModelClient(device=self.device, low_vram=self.low_vram)
        self.depth_handle: DepthHandle = self.model_client.build_depth_model()
        self.mode = mode
        self.resolution = resolution

        if mode == "t2s":
            self.pano_handle: PanoGenHandle = self.model_client.build_pano_gen_model(lora_path=lora_path, mode="t2s")
        elif mode == "i2s":
            self.pano_handle: PanoGenHandle = self.model_client.build_pano_gen_model(lora_path=lora_path, mode="i2s")
        else:
            raise ValueError(f"Invalid mode: {mode}, mode must be 't2s' or 'i2s'")

        self.use_sharp = use_sharp
        self.sharp_handle: Optional[SharpHandle] = None
        if use_sharp:
            self.sharp_handle = self.model_client.build_sharp_model()

        self.inpaint_bg = inpaint_bg
        self.seg_handle: Optional[SegHandle] = None
        self.inpaint_handle: Optional[InpaintHandle] = None
        if inpaint_bg:
            self.seg_handle = self.model_client.build_segment_model()
            self.inpaint_handle = self.model_client.build_inpaint_model()

    def depth2gs(self, predictions: DepthPrediction) -> SplatFile:
        return convert_rgbd_to_gs(predictions.rgb, predictions.distance, predictions.rays)

    def depth2mesh(self, predictions: DepthPrediction) -> o3d.geometry.TriangleMesh:
        return convert_rgbd2mesh_panorama(predictions.rgb / 255.0, predictions.distance, predictions.rays)

    def inpaint_bg_splat(
        self, pano_image: Image.Image, init_splat: SplatFile, init_pred: DepthPrediction
    ) -> SplatFile:
        fg_mask = self.seg_handle.fg_mask(pano_image, init_pred.distance)
        edge_mask = (
            cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
            - cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
        )
        init_splat = mask_splat(init_splat, (1 - edge_mask))

        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=10)
        pano_bg = self.inpaint_handle.inpaint(pano_image, dilated_fg_mask)
        bg_pred = self.depth_handle.predict(pano_bg)
        bg_pred = depth_match(init_pred, bg_pred, (1 - dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = mask_splat(pano_bg_splat, dilated_fg_mask)
        return merge_splats(init_splat, occ_bg_splat)

    def _generate_world(
        self, pano_image: Image.Image, return_mesh: bool = False
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        init_pred = self.depth_handle.predict(pano_image)

        if self.use_sharp:
            return self.sharp_handle.predict(pano_image, init_pred)

        if return_mesh:
            return self.depth2mesh(init_pred)

        splat = self.depth2gs(init_pred)
        if self.inpaint_bg:
            splat = self.inpaint_bg_splat(pano_image, splat, init_pred)
        return splat

    def generate_pano(self, prompt: str = "", image: Optional[Image.Image] = None) -> Image.Image:
        if self.mode == "t2s":
            assert image is None, "image is not supported for text-to-scene generation"
            return self.pano_handle.generate(prompt=prompt, height=self.resolution // 2, width=self.resolution)

        assert image is not None, "image is required for image-to-scene generation"
        image = resize_img(image)
        predictions = self.depth_handle.predict(image)
        pano_cond_img, cond_mask = map_image_to_pano(predictions, device=self.device)
        pano_image = self.pano_handle.fill(
            image=pano_cond_img,
            mask=cond_mask,
            prompt=prompt,
            height=self.resolution // 2,
            width=self.resolution,
        )

        map_height, map_width = pano_cond_img.height, pano_cond_img.width
        pano_image = pano_image.resize((map_width, map_height))
        pano_cond_np = np.array(pano_cond_img)
        mask_np = np.array(cond_mask) / 255.0
        pano_image = np.array(pano_image) * mask_np[:, :, None] + pano_cond_np * (1 - mask_np[:, :, None])
        return Image.fromarray(pano_image.astype(np.uint8))

    @torch.inference_mode()
    def generate_world(
        self,
        prompt: str = "",
        image: Optional[Image.Image] = None,
        return_mesh: bool = False,
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        pano_image = self.generate_pano(prompt, image)
        return self._generate_world(pano_image, return_mesh)
