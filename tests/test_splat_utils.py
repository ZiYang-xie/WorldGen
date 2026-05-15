"""Tests for src/auroch_syna/worldgen/utils/splat_utils.py (CPU-only)."""
import sys
import os
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.utils.splat_utils import SplatFile, mask_splat, merge_splats
from auroch_syna.worldgen.utils.general_utils import pano_unit_rays


def _make_splat(n: int) -> SplatFile:
    rng = np.random.default_rng(0)
    return SplatFile(
        centers=rng.random((n, 3)).astype(np.float32),
        rgbs=rng.random((n, 3)).astype(np.float32),
        opacities=rng.random((n, 1)).astype(np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        scales=rng.random((n, 3)).astype(np.float32),
    )


class TestSplatFileSaveLoad:
    def test_save_creates_file(self):
        splat = _make_splat(10)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        splat.save(path)
        assert os.path.getsize(path) > 0
        os.unlink(path)

    def test_save_valid_ply(self):
        """PLY file should be parseable by plyfile."""
        from plyfile import PlyData
        splat = _make_splat(5)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        splat.save(path)
        data = PlyData.read(path)
        assert "vertex" in [e.name for e in data.elements]
        assert len(data["vertex"]) == 5
        os.unlink(path)


class TestMergeSplats:
    def test_count_doubles(self):
        s1 = _make_splat(10)
        s2 = _make_splat(15)
        merged = merge_splats(s1, s2)
        assert merged.centers.shape[0] == 25

    def test_all_fields_concatenated(self):
        s1 = _make_splat(4)
        s2 = _make_splat(6)
        merged = merge_splats(s1, s2)
        assert merged.rgbs.shape == (10, 3)
        assert merged.opacities.shape == (10, 1)
        assert merged.covariances.shape == (10, 3, 3)
        assert merged.rotations.shape == (10, 4)
        assert merged.scales.shape == (10, 3)


class TestMaskSplat:
    def test_keeps_only_masked_region(self):
        H, W = 4, 8
        # Build a fake splat with H*W points
        n = H * W
        splat = _make_splat(n)
        # Mask: only top-left 2x4 quadrant
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:2, :4] = 1
        result = mask_splat(splat, mask)
        assert result.centers.shape[0] == 8  # 2*4

    def test_zero_mask_returns_empty(self):
        H, W = 2, 4
        splat = _make_splat(H * W)
        mask = np.zeros((H, W), dtype=np.uint8)
        result = mask_splat(splat, mask)
        assert result.centers.shape[0] == 0
