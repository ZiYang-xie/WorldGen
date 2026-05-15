"""Tests for pano_seg.py validation logic (no model required)."""
import sys
import os

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.pano_seg import seg_pano


class TestSegPanoValidation:
    def test_non_panorama_raises(self):
        """seg_pano should raise ValueError for non-2:1 images."""
        img = Image.new("RGB", (100, 100))  # 1:1 ratio
        with pytest.raises(ValueError, match="2:1"):
            seg_pano(None, None, img)

    def test_exact_2_1_ratio_passes_check(self):
        """A 2:1 image should pass the ratio check (will fail later without model)."""
        img = Image.new("RGB", (200, 100))
        # Should NOT raise ValueError for aspect ratio (will fail at model call)
        try:
            seg_pano(None, None, img)
        except ValueError as e:
            if "2:1" in str(e):
                pytest.fail(f"Should not raise aspect ratio error: {e}")
        except Exception:
            pass  # Any other error (AttributeError on None model) is fine
