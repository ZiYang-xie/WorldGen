import threading
from typing import Optional

class ModelClient:
    """Thin wrapper that centralizes model construction/loading.

    Purpose: provide a single boundary so model loading can later be
    redirected to a service or subprocess without changing callers.
    """
    def __init__(self, device: Optional[str] = "cuda", low_vram: bool = False):
        self.device = device
        self.low_vram = low_vram
        self._lock = threading.Lock()
        self._cache = {}

    def build_depth_model(self):
        if "depth" in self._cache:
            return self._cache["depth"]
        with self._lock:
            if "depth" in self._cache:
                return self._cache["depth"]
            from .pano_depth import build_depth_model as _build
            m = _build(device=self.device)
            self._cache["depth"] = m
            return m

    def build_pano_gen_model(self, lora_path=None, mode="t2s"):
        key = f"pano_{mode}_{lora_path}"
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            from .pano_gen import build_pano_gen_model, build_pano_fill_model
            if mode == "t2s":
                m = build_pano_gen_model(lora_path=lora_path, device=self.device, low_vram=self.low_vram)
            else:
                m = build_pano_fill_model(lora_path=lora_path, device=self.device, low_vram=self.low_vram)
            self._cache[key] = m
            return m

    def build_segment_model(self):
        if "seg" in self._cache:
            return self._cache["seg"]
        with self._lock:
            if "seg" in self._cache:
                return self._cache["seg"]
            from .pano_seg import build_segment_model as _build
            m = _build(device=self.device)
            self._cache["seg"] = m
            return m

    def build_inpaint_model(self):
        if "inpaint" in self._cache:
            return self._cache["inpaint"]
        with self._lock:
            if "inpaint" in self._cache:
                return self._cache["inpaint"]
            from .pano_inpaint import build_inpaint_model as _build
            m = _build(device=self.device)
            self._cache["inpaint"] = m
            return m

    def build_sharp_model(self):
        if "sharp" in self._cache:
            return self._cache["sharp"]
        with self._lock:
            if "sharp" in self._cache:
                return self._cache["sharp"]
            from .pano_sharp import build_sharp_model as _build
            m = _build(device=self.device)
            self._cache["sharp"] = m
            return m
