"""Tests for src/auroch_syna/worldgen/utils/general_utils.py (CPU-only)."""
import math
import sys
import os

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.utils.general_utils import (
    pano_unit_rays,
    resize_img,
)


class TestPanoUnitRays:
    def test_shape(self):
        rays = pano_unit_rays(8, 16, torch.device("cpu"))
        assert rays.shape == (8, 16, 3)

    def test_unit_length(self):
        rays = pano_unit_rays(16, 32, torch.device("cpu"))
        norms = torch.linalg.norm(rays, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_covers_full_sphere(self):
        # Mean of unit rays on a full sphere should be close to zero
        rays = pano_unit_rays(64, 128, torch.device("cpu"))
        mean = rays.mean(dim=(0, 1))
        assert mean.abs().max() < 0.05


class TestResizeImg:
    def test_landscape_long_edge_capped(self):
        img = Image.new("RGB", (2000, 800))
        out = resize_img(img, max_size=1024)
        assert max(out.size) == 1024

    def test_portrait_long_edge_capped(self):
        img = Image.new("RGB", (600, 1200))
        out = resize_img(img, max_size=512)
        assert max(out.size) == 512

    def test_small_image_unchanged(self):
        img = Image.new("RGB", (100, 50))
        out = resize_img(img, max_size=1024)
        assert out.size == (100, 50)

    def test_aspect_ratio_preserved(self):
        img = Image.new("RGB", (800, 400))
        out = resize_img(img, max_size=400)
        w, h = out.size
        assert abs(w / h - 2.0) < 0.01
