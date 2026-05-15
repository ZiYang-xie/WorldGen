"""Model factory + transport seam.

The point of this module is to be the only place that knows about
*model construction*. Out-of-process backends (the daemon, a remote
inference server) can implement the same surface area without
touching callers.

Today every caller still consumes the raw pipeline object that
``build_*`` returns. The migration path is:

  1. Continue to return raw pipelines from ``build_*`` (this file).
  2. Introduce ``handle_*`` methods that return ``ModelHandle`` wrappers
     and an associated inference function.
  3. Migrate callers one-by-one to call ``handle.infer(...)`` instead of
     calling the pipeline directly.

This file already implements steps 1 and 2; step 3 is per-call-site.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from auroch_syna.runtime import RuntimePolicy, get_logger
from auroch_syna.runtime.transport import InProcessHandle, ModelHandle

log = get_logger(__name__)


class ModelClient:
    """Centralized model construction with optional policy-driven config."""

    def __init__(
        self,
        device: Optional[str] = "cuda",
        low_vram: bool = False,
        *,
        policy: Optional[RuntimePolicy] = None,
    ) -> None:
        self.device = device
        self.low_vram = low_vram
        self.policy = policy
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    # ---- raw pipeline builders (legacy callers still depend on these) ----

    def build_depth_model(self):
        return self._cached("depth", self._build_depth)

    def build_pano_gen_model(self, lora_path=None, mode="t2s"):
        key = f"pano_{mode}_{lora_path}"
        return self._cached(key, lambda: self._build_pano_gen(lora_path, mode))

    def build_segment_model(self):
        return self._cached("seg", self._build_segment)

    def build_inpaint_model(self):
        return self._cached("inpaint", self._build_inpaint)

    def build_sharp_model(self):
        return self._cached("sharp", self._build_sharp)

    # ---- ModelHandle accessors (preferred for new callers) ----

    def handle_depth(self) -> ModelHandle:
        from .pano_depth import pred_pano_depth

        return InProcessHandle(
            name="depth",
            pipeline=self.build_depth_model(),
            infer_fn=pred_pano_depth,
        )

    def handle_pano_gen(self, lora_path=None, mode="t2s") -> ModelHandle:
        from .pano_gen import gen_pano_fill_image, gen_pano_image

        pipeline = self.build_pano_gen_model(lora_path=lora_path, mode=mode)
        infer_fn = gen_pano_image if mode == "t2s" else gen_pano_fill_image
        return InProcessHandle(name=f"pano_gen_{mode}", pipeline=pipeline, infer_fn=infer_fn)

    def handle_segment(self) -> ModelHandle:
        from .pano_seg import seg_pano

        processor, model = self.build_segment_model()
        # Wrap (processor, model) as a single pipeline tuple.
        def _infer(pkg, image):
            proc, mdl = pkg
            return seg_pano(proc, mdl, image)

        return InProcessHandle(name="segment", pipeline=(processor, model), infer_fn=_infer)

    def handle_inpaint(self) -> ModelHandle:
        from .pano_inpaint import inpaint_image

        return InProcessHandle(
            name="inpaint",
            pipeline=self.build_inpaint_model(),
            infer_fn=inpaint_image,
        )

    def handle_sharp(self) -> ModelHandle:
        from .pano_sharp import predict_equirectangular

        return InProcessHandle(
            name="sharp",
            pipeline=self.build_sharp_model(),
            infer_fn=predict_equirectangular,
        )

    # ---- helpers ----

    def _cached(self, key: str, builder):
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            built = builder()
            self._cache[key] = built
            return built

    def _build_depth(self):
        from .pano_depth import build_depth_model
        return build_depth_model(device=self.device)

    def _build_pano_gen(self, lora_path, mode):
        from .pano_gen import build_pano_fill_model, build_pano_gen_model

        torch_dtype = self.policy.torch_dtype() if self.policy else None
        enable_offload = self.policy.enable_cpu_offload if self.policy else None
        builder = build_pano_gen_model if mode == "t2s" else build_pano_fill_model
        return builder(
            lora_path=lora_path,
            device=self.device,
            low_vram=self.low_vram,
            torch_dtype=torch_dtype,
            enable_cpu_offload=enable_offload,
        )

    def _build_segment(self):
        from .pano_seg import build_segment_model
        return build_segment_model(device=self.device)

    def _build_inpaint(self):
        from .pano_inpaint import build_inpaint_model
        return build_inpaint_model(device=self.device)

    def _build_sharp(self):
        from .pano_sharp import build_sharp_model
        return build_sharp_model(device=self.device)
