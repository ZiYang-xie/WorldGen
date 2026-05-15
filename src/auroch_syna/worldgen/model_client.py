import threading
import torch
from typing import Optional

from .device_utils import resolve_device
from .model_handle import DepthHandle, PanoGenHandle, SegHandle, InpaintHandle, SharpHandle


class ModelClient:
    """Thin wrapper that centralizes model construction/loading.

    Purpose: provide a single boundary so model loading can later be
    redirected to a service or subprocess without changing callers.
    Cache keys include device and low_vram so that two clients with
    different configurations never share a cached model.

    All build_* methods return typed handle objects rather than raw models.
    """

    def __init__(self, device: Optional[str | torch.device] = None, low_vram: bool = False):
        self.device = resolve_device(device)
        self.low_vram = low_vram
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._key_prefix = f"{self.device}:low_vram={low_vram}"

    def _load(self, key: str, factory):
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            obj = factory()
            self._cache[key] = obj
            return obj

    def build_depth_model(self) -> DepthHandle:
        def factory():
            from .pano_depth import build_depth_model as _build
            return DepthHandle(_build(device=self.device))
        return self._load(f"{self._key_prefix}:depth", factory)

    def build_pano_gen_model(self, lora_path=None, mode: str = "t2s") -> PanoGenHandle:
        def factory():
            from .pano_gen import build_pano_gen_model, build_pano_fill_model
            if mode == "t2s":
                pipe = build_pano_gen_model(lora_path=lora_path, device=self.device, low_vram=self.low_vram)
            else:
                pipe = build_pano_fill_model(lora_path=lora_path, device=self.device, low_vram=self.low_vram)
            return PanoGenHandle(pipe, mode)
        return self._load(f"{self._key_prefix}:pano_{mode}_{lora_path}", factory)

    def build_segment_model(self) -> SegHandle:
        def factory():
            from .pano_seg import build_segment_model as _build
            processor, model = _build(device=self.device)
            return SegHandle(processor, model)
        return self._load(f"{self._key_prefix}:seg", factory)

    def build_inpaint_model(self) -> InpaintHandle:
        def factory():
            from .pano_inpaint import build_inpaint_model as _build
            return InpaintHandle(_build(device=self.device))
        return self._load(f"{self._key_prefix}:inpaint", factory)

    def build_sharp_model(self) -> SharpHandle:
        def factory():
            from .pano_sharp import build_sharp_model as _build
            return SharpHandle(_build(device=self.device), self.device)
        return self._load(f"{self._key_prefix}:sharp", factory)
