"""Typed handles wrapping raw models with their inference functions.

Each handle owns one model (or model pair) and exposes domain-level methods,
so callers never import module-level inference helpers directly.
"""
from __future__ import annotations

import torch
from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .pano_depth import DepthPrediction
    from .utils.splat_utils import SplatFile


class DepthHandle:
    """Wraps the DA-2 depth model."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, image: Image.Image) -> "DepthPrediction":
        from .pano_depth import pred_depth
        return pred_depth(self._model, image)


class PanoGenHandle:
    """Wraps a FLUX panorama generation pipeline (text-to-scene or fill)."""

    def __init__(self, pipe, mode: str) -> None:
        self._pipe = pipe
        self._mode = mode

    def generate(self, prompt: str = "", **kwargs) -> Image.Image:
        """Generate a panorama from text (t2s mode)."""
        from .pano_gen import gen_pano_image
        return gen_pano_image(self._pipe, prompt=prompt, **kwargs)

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str = "", **kwargs) -> Image.Image:
        """Fill/inpaint a masked panorama conditioned on an image (i2s mode)."""
        from .pano_gen import gen_pano_fill_image
        return gen_pano_fill_image(self._pipe, image=image, mask=mask, prompt=prompt, **kwargs)


class SegHandle:
    """Wraps the OneFormer segmentation model + processor pair."""

    def __init__(self, processor, model) -> None:
        self._processor = processor
        self._model = model

    def fg_mask(self, image: Image.Image, depth: torch.Tensor) -> "np.ndarray":
        """Return a foreground mask derived from semantics and depth."""
        from .pano_seg import seg_pano_fg
        return seg_pano_fg(self._processor, self._model, image, depth)


class InpaintHandle:
    """Wraps the LaMa inpainting model."""

    def __init__(self, pipe) -> None:
        self._pipe = pipe

    def inpaint(self, image: Image.Image, mask: "np.ndarray") -> Image.Image:
        """Inpaint masked regions in a panorama."""
        from .pano_inpaint import inpaint_pano
        return inpaint_pano(self._pipe, image, mask)


class SharpHandle:
    """Wraps the ML-Sharp RGB-Gaussian predictor."""

    def __init__(self, model, device: torch.device) -> None:
        self._model = model
        self._device = device

    def predict(
        self,
        pano: Image.Image,
        depth_predictions: "DepthPrediction",
        face_size: int | None = None,
    ) -> "SplatFile":
        """Predict Gaussians from an equirectangular image via cubemap decomposition."""
        from .pano_sharp import predict_equirectangular
        from .constants import DEFAULT_CUBEMAP_FACE_SIZE
        kwargs = {} if face_size is None else {"face_size": face_size}
        return predict_equirectangular(
            self._model, pano, self._device,
            depth_predictions=depth_predictions,
            **kwargs,
        )
