import json
from typing import Any, Dict, Optional

import requests


class ModelClientRPC:
    """Minimal HTTP RPC client for ModelClient-backed model services.

    This is intentionally tiny and dependency-light; it sends JSON payloads to
    a local mock server during development. Methods mirror the high-level
    model operations used in `WorldGen`.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/') }"
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def ping(self) -> Dict[str, Any]:
        return self._post("ping", {"msg": "ping"})

    def predict_depth(self, image_b64: str) -> Dict[str, Any]:
        return self._post("predict_depth", {"image_b64": image_b64})

    def generate_pano(self, prompt: str, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._post("generate_pano", {"prompt": prompt, "seed": seed})

    def inpaint(self, image_b64: str, mask_b64: str) -> Dict[str, Any]:
        return self._post("inpaint", {"image_b64": image_b64, "mask_b64": mask_b64})

    def segment(self, image_b64: str) -> Dict[str, Any]:
        return self._post("segment", {"image_b64": image_b64})

    def sharp(self, pano_b64: str) -> Dict[str, Any]:
        return self._post("sharp", {"pano_b64": pano_b64})


__all__ = ["ModelClientRPC"]
