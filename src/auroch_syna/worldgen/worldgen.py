"""High-level orchestrator that coordinates the text/image → 3D scene pipeline."""
from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from auroch_syna.runtime import (
    EventBus,
    RuntimePolicy,
    default_bus,
    get_logger,
    select_policy,
)

from .model_client import ModelClient
from .pano_depth import pred_depth, pred_pano_depth
from .pano_gen import gen_pano_fill_image, gen_pano_image
from .utils.general_utils import (
    convert_rgbd2mesh_panorama,
    depth_match,
    map_image_to_pano,
    resize_img,
)
from .utils.splat_utils import (
    SplatFile,
    convert_rgbd_to_gs,
    mask_splat,
    merge_splats,
)

log = get_logger(__name__)


class WorldGen:
    def __init__(
        self,
        mode: str = "t2s",
        use_sharp: bool = False,
        inpaint_bg: bool = False,
        lora_path: Optional[str] = None,
        resolution: int = 1600,
        device: Optional[Union[str, torch.device]] = None,
        low_vram: Optional[bool] = None,
        *,
        bus: Optional[EventBus] = None,
        policy: Optional[RuntimePolicy] = None,
    ):
        """Construct a WorldGen orchestrator.

        ``device`` accepts: ``None`` / ``"auto"`` (let the runtime pick),
        ``"cuda"``, ``"mps"``, ``"cpu"``, or any ``torch.device``.
        """
        self.bus = bus or default_bus()

        if policy is None:
            prefer = device if isinstance(device, str) else (str(device) if device is not None else None)
            policy = select_policy(prefer_device=prefer, explicit_low_vram=low_vram)
        self.policy = policy
        self.device = policy.device
        for note in policy.notes:
            log.info("policy: %s", note)
        log.info(
            "Resolved policy: device=%s dtype=%s backend=%s low_vram=%s offload=%s",
            policy.device, policy.dtype, policy.backend, policy.low_vram, policy.enable_cpu_offload,
        )
        self._publish("pipeline.policy", {
            "device": policy.device,
            "dtype": policy.dtype,
            "backend": policy.backend,
            "low_vram": policy.low_vram,
        })

        self.low_vram = policy.low_vram
        self.mode = mode
        self.resolution = resolution
        self.use_sharp = use_sharp
        self.inpaint_bg = inpaint_bg

        self.model_client = ModelClient(device=self.device, low_vram=self.low_vram, policy=policy)
        self._publish("pipeline.stage", {"stage": "build_depth_model"})
        self.depth_model = self.model_client.build_depth_model()

        if mode not in ("t2s", "i2s"):
            raise ValueError(f"Invalid mode: {mode}, mode must be 't2s' or 'i2s'")
        self._publish("pipeline.stage", {"stage": "build_pano_gen_model", "mode": mode})
        self.pano_gen_model = self.model_client.build_pano_gen_model(lora_path=lora_path, mode=mode)

        if use_sharp:
            self._publish("pipeline.stage", {"stage": "build_sharp_model"})
            self.sharp_model = self.model_client.build_sharp_model()

        if inpaint_bg:
            self._publish("pipeline.stage", {"stage": "build_segment_model"})
            self.seg_processor, self.seg_model = self.model_client.build_segment_model()
            self._publish("pipeline.stage", {"stage": "build_inpaint_model"})
            self.inpaint_pipe = self.model_client.build_inpaint_model()

    def _publish(self, kind: str, payload: dict) -> None:
        self.bus.publish(kind, payload)

    def depth2gs(self, predictions) -> SplatFile:
        return convert_rgbd_to_gs(
            predictions["rgb"], predictions["distance"], predictions["rays"]
        )

    def depth2mesh(self, predictions) -> o3d.geometry.TriangleMesh:
        return convert_rgbd2mesh_panorama(
            predictions["rgb"] / 255.0, predictions["distance"], predictions["rays"]
        )

    def inpaint_bg_splat(
        self, pano_image: Image.Image, init_splat: SplatFile, init_pred: dict
    ) -> SplatFile:
        from .pano_inpaint import inpaint_image
        from .pano_seg import seg_pano_fg

        fg_mask = seg_pano_fg(self.seg_processor, self.seg_model, pano_image, init_pred["distance"])
        edge_mask = (
            cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
            - cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
        )
        init_splat = mask_splat(init_splat, (1 - edge_mask))

        dilated_fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=10)
        pano_bg = inpaint_image(self.inpaint_pipe, pano_image, dilated_fg_mask)
        bg_pred = pred_pano_depth(self.depth_model, pano_bg)
        bg_pred = depth_match(init_pred, bg_pred, (1 - dilated_fg_mask))
        pano_bg_splat = self.depth2gs(bg_pred)
        occ_bg_splat = mask_splat(pano_bg_splat, dilated_fg_mask)
        return merge_splats(init_splat, occ_bg_splat)

    def _generate_world(
        self, pano_image: Image.Image, return_mesh: bool = False
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        self._publish("pipeline.stage", {"stage": "depth"})
        init_pred = pred_pano_depth(self.depth_model, pano_image)

        if self.use_sharp:
            from .pano_sharp import predict_equirectangular

            self._publish("pipeline.stage", {"stage": "sharp_splats"})
            return predict_equirectangular(
                self.sharp_model, pano_image, device=self.device, depth_predictions=init_pred
            )

        if return_mesh:
            self._publish("pipeline.stage", {"stage": "mesh"})
            return self.depth2mesh(init_pred)

        self._publish("pipeline.stage", {"stage": "splats"})
        splat = self.depth2gs(init_pred)
        if self.inpaint_bg:
            self._publish("pipeline.stage", {"stage": "inpaint_bg"})
            splat = self.inpaint_bg_splat(pano_image, splat, init_pred)
        return splat

    def generate_pano(self, prompt: str = "", image: Optional[Image.Image] = None) -> Image.Image:
        if self.mode == "t2s":
            assert image is None, "image is not supported for text-to-scene generation"
            self._publish("pipeline.stage", {"stage": "gen_pano_t2s", "prompt": prompt})
            return gen_pano_image(
                self.pano_gen_model,
                prompt=prompt,
                height=self.resolution // 2,
                width=self.resolution,
            )

        assert image is not None, "image is required for image-to-scene generation"
        self._publish("pipeline.stage", {"stage": "gen_pano_i2s", "prompt": prompt})
        image = resize_img(image)
        predictions = pred_depth(self.depth_model, image)
        pano_cond_img, cond_mask = map_image_to_pano(predictions)
        pano_image = gen_pano_fill_image(
            self.pano_gen_model,
            image=pano_cond_img,
            mask=cond_mask,
            prompt=prompt,
            height=self.resolution // 2,
            width=self.resolution,
        )

        map_height, map_width = pano_cond_img.height, pano_cond_img.width
        pano_image = pano_image.resize((map_width, map_height))
        pano_cond_img, mask = np.array(pano_cond_img), np.array(cond_mask) / 255.0
        pano_image = np.array(pano_image) * mask[:, :, None] + pano_cond_img * (1 - mask[:, :, None])
        return Image.fromarray(pano_image.astype(np.uint8))

    @torch.inference_mode()
    def generate_world(
        self,
        prompt: str = "",
        image: Optional[Image.Image] = None,
        return_mesh: bool = False,
    ) -> Union[SplatFile, o3d.geometry.TriangleMesh]:
        self._publish("pipeline.start", {"prompt": prompt, "has_image": image is not None,
                                          "return_mesh": return_mesh})
        try:
            pano_image = self.generate_pano(prompt, image)
            scene = self._generate_world(pano_image, return_mesh)
        except Exception as e:
            self._publish("pipeline.error", {"error": str(e), "type": type(e).__name__})
            raise
        self._publish("pipeline.complete", {})
        return scene
