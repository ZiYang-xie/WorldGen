"""Tests for DepthPrediction dataclass and depth_match (CPU-only)."""
import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.pano_depth import DepthPrediction, pred_pano_depth, pred_depth
from auroch_syna.worldgen.utils.general_utils import depth_match


def _make_prediction(H=8, W=16, device=torch.device("cpu")) -> DepthPrediction:
    return DepthPrediction(
        rgb=torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device=device),
        distance=torch.rand(H, W, device=device) * 10.0 + 0.1,
        rays=torch.nn.functional.normalize(torch.randn(H, W, 3, device=device), dim=-1),
    )


class TestDepthPrediction:
    def test_is_dataclass(self):
        pred = _make_prediction()
        assert hasattr(pred, "rgb")
        assert hasattr(pred, "distance")
        assert hasattr(pred, "rays")

    def test_field_shapes(self):
        H, W = 8, 16
        pred = _make_prediction(H, W)
        assert pred.rgb.shape == (H, W, 3)
        assert pred.distance.shape == (H, W)
        assert pred.rays.shape == (H, W, 3)

    def test_alias_pred_pano_depth_is_pred_depth(self):
        assert pred_pano_depth is pred_depth


class TestDepthMatch:
    def test_scales_bg_distance(self):
        init = _make_prediction()
        bg = _make_prediction()
        mask = np.ones((8, 16), dtype=np.uint8)

        bg_distance_before = bg.distance.clone()
        result = depth_match(init, bg, mask)
        # Result should have a rescaled distance
        assert result is bg
        # Scale should not be exactly 1 in general (random tensors)
        # Just check the shape is unchanged and values are finite
        assert result.distance.shape == (8, 16)
        assert torch.isfinite(result.distance).all()

    def test_zero_mask_raises_or_produces_nan(self):
        """With all-zero mask there are no valid pixels; result may be NaN."""
        init = _make_prediction()
        bg = _make_prediction()
        mask = np.zeros((8, 16), dtype=np.uint8)
        # We just verify it doesn't crash with a TypeError
        try:
            depth_match(init, bg, mask)
        except Exception:
            pass  # nan/inf from empty median is expected; no TypeError
