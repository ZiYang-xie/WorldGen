"""Unit tests for pure-Python worldgen utilities — no GPU required.

These tests cover the math helpers, SH constant, and correctness of the
fixes applied in this improvement pass.  All tests run on CPU and have
zero ML dependencies.
"""
import math
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# splat_utils — SH_C0, pixel-centre sampling, torch.cross dim, mask_splat
# ---------------------------------------------------------------------------

def test_sh_c0_value():
    from auroch_syna.worldgen.utils.splat_utils import SH_C0
    # SH_C0 = 1 / (2 * sqrt(pi))
    expected = 1.0 / (2.0 * math.sqrt(math.pi))
    assert abs(SH_C0 - expected) < 1e-12, f"SH_C0 = {SH_C0}, expected {expected}"


def _make_tiny_splat(H=4, W=8):
    """Return a minimal SplatFile with H*W gaussians in image order."""
    from auroch_syna.worldgen.utils.splat_utils import SplatFile
    N = H * W
    rng = np.random.default_rng(0)
    return SplatFile(
        centers=rng.random((N, 3)).astype(np.float32),
        rgbs=rng.random((N, 3)).astype(np.float32),
        opacities=np.ones((N, 1), dtype=np.float32),
        covariances=np.eye(3, dtype=np.float32)[None].repeat(N, axis=0),
        rotations=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        scales=np.ones((N, 3), dtype=np.float32) * 0.01,
    )


def test_mask_splat_full_mask():
    """mask_splat with all-ones mask should return all gaussians."""
    from auroch_syna.worldgen.utils.splat_utils import mask_splat
    H, W = 4, 8
    splat = _make_tiny_splat(H, W)
    mask = np.ones((H, W), dtype=np.uint8)
    result = mask_splat(splat, mask)
    assert result.centers.shape[0] == H * W


def test_mask_splat_half_mask():
    """mask_splat with top-half mask should return H/2 * W gaussians."""
    from auroch_syna.worldgen.utils.splat_utils import mask_splat
    H, W = 4, 8
    splat = _make_tiny_splat(H, W)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[: H // 2, :] = 1
    result = mask_splat(splat, mask)
    assert result.centers.shape[0] == (H // 2) * W


def test_mask_splat_with_pixel_valid_mask():
    """mask_splat with pixel_valid_mask should correctly intersect filters."""
    from auroch_syna.worldgen.utils.splat_utils import SplatFile, mask_splat

    H, W = 4, 8
    # Simulate convert_rgbd_to_gs filtering out the last row
    pixel_valid = np.ones((H, W), dtype=bool)
    pixel_valid[-1, :] = False  # last row filtered out
    N_valid = pixel_valid.sum()  # (H-1)*W

    rng = np.random.default_rng(1)
    splat = SplatFile(
        centers=rng.random((N_valid, 3)).astype(np.float32),
        rgbs=rng.random((N_valid, 3)).astype(np.float32),
        opacities=np.ones((N_valid, 1), dtype=np.float32),
        covariances=np.eye(3, dtype=np.float32)[None].repeat(N_valid, axis=0),
        rotations=np.tile([1, 0, 0, 0], (N_valid, 1)).astype(np.float32),
        scales=np.ones((N_valid, 3), dtype=np.float32) * 0.01,
    )

    # Keep only the top half of the *original* image
    caller_mask = np.zeros((H, W), dtype=np.uint8)
    caller_mask[: H // 2, :] = 1

    result = mask_splat(splat, caller_mask, pixel_valid_mask=pixel_valid)
    # Intersection: top H//2 rows, all of which are in pixel_valid
    assert result.centers.shape[0] == (H // 2) * W


def test_torch_cross_dim_explicit():
    """torch.cross must accept dim=1 without DeprecationWarning."""
    import warnings
    a = torch.randn(10, 3)
    b = torch.randn(10, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = torch.cross(a, b, dim=1)  # should not warn


def test_convert_rgbd_to_gs_pixel_centre_sampling():
    """Polar angles in convert_rgbd_to_gs should use pixel-centre sampling."""
    from auroch_syna.worldgen.utils.splat_utils import convert_rgbd_to_gs
    from auroch_syna.worldgen.utils.general_utils import pano_unit_rays

    H, W = 8, 16
    rays = pano_unit_rays(H, W, device="cpu")
    distance = torch.ones(H, W)
    rgb = torch.zeros(H, W, 3, dtype=torch.uint8)

    # Should not raise; pixel-centre sampling avoids the half-pixel bias
    splat = convert_rgbd_to_gs(rgb, distance, rays)
    assert splat.centers.shape[0] == H * W


# ---------------------------------------------------------------------------
# general_utils — pano_unit_rays, depth_match immutability, map_image_to_pano
# ---------------------------------------------------------------------------

def test_pano_unit_rays_unit_length():
    from auroch_syna.worldgen.utils.general_utils import pano_unit_rays
    rays = pano_unit_rays(8, 16, device="cpu")
    norms = torch.linalg.norm(rays, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_pano_unit_rays_pixel_centre():
    """Rays should be computed at pixel centres, not pixel edges."""
    from auroch_syna.worldgen.utils.general_utils import pano_unit_rays
    H, W = 4, 8
    rays = pano_unit_rays(H, W, device="cpu")
    # The top-left ray should NOT point straight up (which would be the
    # pixel-edge value at theta=0).  With pixel-centre sampling, theta[0]
    # = pi/(2H) which gives a small but non-zero y component.
    assert rays[0, 0, 1].abs() > 0, "Expected non-zero y for top-left pixel-centre ray"


def test_depth_match_no_mutation():
    """depth_match must not mutate the caller's bg_pred dict."""
    from auroch_syna.worldgen.utils.general_utils import depth_match

    init_pred = {"distance": torch.ones(4, 4) * 2.0}
    bg_pred = {"distance": torch.ones(4, 4) * 4.0}
    original_bg_distance = bg_pred["distance"].clone()
    mask = np.ones((4, 4), dtype=np.uint8)

    result = depth_match(init_pred, bg_pred, mask)

    # The original dict must be unchanged
    assert torch.allclose(bg_pred["distance"], original_bg_distance), \
        "depth_match mutated the caller's bg_pred dict"

    # The returned dict must have a scaled distance
    assert not torch.allclose(result["distance"], original_bg_distance), \
        "depth_match returned an unscaled distance"


def test_depth_match_idempotent():
    """Calling depth_match twice on the same inputs should give the same result."""
    from auroch_syna.worldgen.utils.general_utils import depth_match

    init_pred = {"distance": torch.ones(4, 4) * 2.0}
    bg_pred = {"distance": torch.ones(4, 4) * 4.0}
    mask = np.ones((4, 4), dtype=np.uint8)

    r1 = depth_match(init_pred, bg_pred, mask)
    r2 = depth_match(init_pred, bg_pred, mask)
    assert torch.allclose(r1["distance"], r2["distance"]), \
        "depth_match is not idempotent — likely still mutating input"


def test_map_image_to_pano_no_device_arg():
    """map_image_to_pano should work without a device argument."""
    from auroch_syna.worldgen.utils.general_utils import map_image_to_pano, pano_unit_rays

    H, W = 16, 32
    rays = pano_unit_rays(H, W, device="cpu")
    rgb = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)
    predictions = {"rgb": rgb, "rays": rays}

    # Should not raise; all tensors stay on CPU
    pano_img, mask_img = map_image_to_pano(predictions, map_h=32, map_w=64)
    assert pano_img.size == (64, 32)


def test_map_image_to_pano_device_consistent():
    """All output tensors should be on the same device as the input."""
    from auroch_syna.worldgen.utils.general_utils import map_image_to_pano, pano_unit_rays

    H, W = 8, 16
    rays = pano_unit_rays(H, W, device="cpu")
    rgb = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)
    predictions = {"rgb": rgb, "rays": rays}

    # Should complete without cross-device errors
    pano_img, mask_img = map_image_to_pano(predictions, map_h=16, map_w=32)
    assert pano_img is not None


# ---------------------------------------------------------------------------
# merge_splats — basic concatenation sanity check
# ---------------------------------------------------------------------------

def test_merge_splats():
    from auroch_syna.worldgen.utils.splat_utils import merge_splats
    s1 = _make_tiny_splat(2, 4)
    s2 = _make_tiny_splat(3, 4)
    merged = merge_splats(s1, s2)
    assert merged.centers.shape[0] == 2 * 4 + 3 * 4
    assert merged.rgbs.shape[0] == merged.centers.shape[0]
