"""Tests for pano_inpaint.py validation logic (no model required)."""
import sys
import os

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.pano_inpaint import inpaint_pano


class TestInpaintPanoValidation:
    def test_non_panorama_raises(self):
        img = Image.new("RGB", (100, 100))  # 1:1
        import numpy as np
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="2:1"):
            inpaint_pano(None, img, mask)

    def test_valid_ratio_passes_check(self):
        img = Image.new("RGB", (200, 100))  # 2:1
        import numpy as np
        mask = np.zeros((100, 200), dtype=np.uint8)
        try:
            inpaint_pano(None, img, mask)
        except ValueError as e:
            if "2:1" in str(e):
                pytest.fail(f"Should not raise aspect ratio error: {e}")
        except Exception:
            pass
