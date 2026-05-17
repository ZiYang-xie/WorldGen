# Auroch Syna (Syna) — Improvement Audit

_Last updated: 2026-05-17_

## Summary

Syna is the local fork of `WorldGen` — a Python ML pipeline that turns a
text prompt or image into a 360° panorama, lifts that panorama to depth +
Gaussian splats (or a triangle mesh), and serves the scene through a
Viser web viewer. This document records concrete, file-grounded findings
and their resolution status. Companion doc `ARCHITECTURE.md` covers the
higher-level migration plan; this file focuses on what was wrong in the
code and what has been fixed.

---

## 0. Tree state and rebrand status

The rebrand from `worldgen` to `auroch_syna` is complete.

| Item | Status |
|------|--------|
| `src/worldgen/` → `src/auroch_syna/worldgen/` | ✅ Done |
| `worldgen.egg-info/` → `src/auroch_syna.egg-info/` | ✅ Done |
| `src/auroch_syna/__init__.py` shim | ✅ Done |
| `ARCHITECTURE.md` added | ✅ Done |
| `.github/workflows/` CI | ✅ Expanded (see §4.4) |
| README fully renamed | ✅ Done |
| `*.before-*` / `*.broken.*` backup files | ✅ Deleted; patterns in `.gitignore` |
| Dead `submodules/pytorch3d` entry in `.gitmodules` | ✅ Removed |
| `submodules/ml-sharp` SSH URL | ✅ Changed to HTTPS |

---

## 1. The `auroch_syna` shim

All issues from the original audit have been resolved.

| Issue | Status |
|-------|--------|
| Mixed tabs/spaces | ✅ Fixed |
| Dead fallback path (`worldgen.utils.splat_utils`) | ✅ Fixed — correct path used |
| Bare `except Exception:` | ✅ Fixed — narrowed to `ImportError` |
| `SplatFile` import broken | ✅ Fixed — correct path `auroch_syna.worldgen.utils.splat_utils` |

---

## 2. Correctness bugs

### 2.1 `flux_pano_fill_pipeline.py` — `NameError` on `latent_image_ids` early-return path

**Status: ✅ Fixed.**

`latent_image_ids` is now computed via `_prepare_latent_image_ids` before
the early return so the return signature is consistent with the normal path.

### 2.2 `lora_utils.py` — silent LoRA dropping on macOS

**Status: ✅ Fixed.**

`compose_lora_with_fixes` now emits `warnings.warn(..., RuntimeWarning)`
when multiple LoRAs are requested but Nunchaku is unavailable, listing the
identities of all dropped LoRAs.

### 2.3 `pano_depth.py` — `pred_pano_depth` and `pred_depth` byte-identical

**Status: ✅ Fixed.**

`pred_depth` is now a documented alias for `pred_pano_depth`.

### 2.4 `pano_seg.py` — debug remnant `['high', 'highest'][0]`

**Status: ✅ Fixed.**

Replaced with `torch.set_float32_matmul_precision("high")`.

### 2.5 Fragile float equality (`pano_seg.py`, `pano_inpaint.py`)

**Status: ✅ Fixed.**

Both files now use `H * 2 == W` (integer arithmetic).

### 2.6 `splat_utils.py` — `mask_splat` assumes unfiltered image order

**Status: ✅ Fixed.**

`mask_splat` now accepts an optional `pixel_valid_mask` keyword argument.
When provided (from the `valid_mask_out` list passed to
`convert_rgbd_to_gs`), the caller's mask is intersected with the filter
applied during splat construction, making the reshape always safe.
`convert_rgbd_to_gs` gained a `valid_mask_out: list | None = None`
parameter to expose this mask to callers.

### 2.7 `splat_utils.py` — deprecated `torch.cross` calls

**Status: ✅ Fixed.**

All three `torch.cross(...)` calls now pass `dim=1` explicitly.

### 2.8 `general_utils.py` — device-mismatch trap in `map_image_to_pano`

**Status: ✅ Fixed.**

The `device` parameter has been removed. All intermediate tensors
(including `rays_pano` and `valid_mask`) are now built on `rgb_src.device`,
eliminating the cross-device indexing crash on MPS/CPU inputs.

### 2.9 `general_utils.py` — `depth_match` silently mutates its input

**Status: ✅ Fixed.**

`depth_match` now returns a shallow copy of `bg_pred` with the `distance`
tensor replaced by a scaled copy. The caller's dict is never mutated.

---

## 3. Portability / device-handling issues

| File | Issue | Status |
|------|-------|--------|
| `pano_depth.py` | `device='cuda'` default | ✅ Fixed — uses `resolve_device()` |
| `pano_seg.py` | `device='cuda'` default | ✅ Fixed — uses `resolve_device()` |
| `pano_inpaint.py` | `device='cuda'` default | ✅ Fixed — uses `resolve_device()` |
| `models/inpaint_model.py` | `LaMa(device='cuda')` | ✅ Fixed |
| `worldgen.py` | `device='cuda'` default | ✅ Fixed — uses `select_policy()` |
| `pano_gen.py` | `device="mps"` default | ✅ Fixed — uses `resolve_device()` |
| `general_utils.py` | `convert_rgbd2mesh_panorama` `device='cuda'` | ✅ Fixed — defaults to `rgb.device` |
| `pano_depth.py` | `torch.autocast` raises on CPU | ✅ Fixed — uses `safe_autocast()` from `runtime.torch_compat` |
| `pano_sharp.py` | Top-level `sharp.*` imports | ✅ Fixed — deferred into function bodies |
| `pano_inpaint.py` | Unconditional CWD save | ✅ Fixed — `save_path: str | None = None` |
| `models/inpaint_model.py` | Third-party LaMa mirror URL | ⚠️ Open — consider mirroring on HuggingFace with sha256 |
| `pano_sharp.py` | Apple CDN URL, no checksum | ⚠️ Open — consider mirroring with sha256 |

---

## 4. Architectural improvements

### 4.1 `ModelClient` / `ModelHandle` protocol

**Status: ✅ Done.**

`runtime/transport.py` defines `ModelHandle` (protocol), `InProcessHandle`,
and a placeholder `RemoteHandle`. `ModelClient` returns `InProcessHandle`
objects from its `handle_*` methods.

### 4.2 LoRA "fix" logic — hardcoded block count and dtype

**Status: ✅ Fixed.**

`load_and_fix_lora` now derives `n_single` and `n_transformer` from the
block indices present in the loaded checkpoint, falling back to 29 only
when no block keys are found. Zero-filled tensors now use the dtype of the
existing LoRA tensors (`zero_dtype = ref_tensor.dtype`) to avoid implicit
float32 upcasts.

### 4.3 FLUX pipelines monkey-patch the VAE

**Status: ⚠️ Open.**

The `_decode` / `tiled_decode` helpers are still defined inside `__call__`
and dynamically bound to `self.vae`. Promoting them to real subclass
methods on a `PanoFluxVAE` mixin is a worthwhile refactor but has no
correctness impact today.

### 4.4 CI — no package install, no ruff, no pytest

**Status: ✅ Fixed.**

`.github/workflows/smoke-import.yml` now contains three jobs:

1. **lint** — installs `ruff` and runs `ruff check src/ tests/`.
2. **unit-tests** — installs `.[dev]` (core package + pytest/ruff/mypy)
   and runs `pytest tests/ -v --tb=short`.
3. **smoke-import** — installs the core package only and verifies both
   importability and that the lazy shim does not eagerly load
   `worldgen.worldgen`.

### 4.5 Diagnostics use `print()` everywhere

**Status: ✅ Fixed.**

All modules use `logging.getLogger(__name__)` via the `get_logger` helper
from `auroch_syna.runtime`.

### 4.6 `worldgen/__init__.py` missing `SplatFile`

**Status: ✅ Fixed.**

`auroch_syna/worldgen/__init__.py` exports both `WorldGen` and `SplatFile`.

### 4.7 Magic constants in `splat_utils.py`

**Status: ✅ Fixed.**

- `SH_C0 = 0.28209479177387814` is now a named module-level constant with
  a comment explaining its mathematical origin.
- Pixel-edge sampling in `convert_rgbd_to_gs` has been unified to
  pixel-centre sampling (`(arange + 0.5) / H * pi`) to match
  `pano_unit_rays` in `general_utils.py`.

### 4.8 `ARCHITECTURE.md` / `REPORT.md` scope overlap

**Status: ✅ Resolved.**

- **REPORT.md** (this file): findings, bugs, resolution status, near-term
  punch list.
- **ARCHITECTURE.md**: target architecture, migration plan, design
  primitives.

Each document links to the other.

---

## 5. New unit tests added

`tests/test_worldgen_utils.py` covers the pure-Python utilities with no
GPU requirement:

| Test | What it verifies |
|------|-----------------|
| `test_sh_c0_value` | `SH_C0 == 1 / (2√π)` to 12 decimal places |
| `test_mask_splat_full_mask` | All gaussians retained with all-ones mask |
| `test_mask_splat_half_mask` | Correct count with top-half mask |
| `test_mask_splat_with_pixel_valid_mask` | Intersection with `pixel_valid_mask` |
| `test_torch_cross_dim_explicit` | No `DeprecationWarning` from `torch.cross` |
| `test_convert_rgbd_to_gs_pixel_centre_sampling` | No crash with pixel-centre theta |
| `test_pano_unit_rays_unit_length` | All rays have unit norm |
| `test_pano_unit_rays_pixel_centre` | Top-left ray has non-zero y (pixel-centre) |
| `test_depth_match_no_mutation` | Caller's `bg_pred` dict is not mutated |
| `test_depth_match_idempotent` | Calling twice gives the same result |
| `test_map_image_to_pano_no_device_arg` | Works without `device=` argument |
| `test_map_image_to_pano_device_consistent` | No cross-device errors on CPU |
| `test_merge_splats` | Concatenation produces correct total count |

---

## 6. Remaining open items (lower priority)

In priority order:

1. **Mirror Sharp + LaMa weights** on HuggingFace with pinned sha256s so
   `pano_sharp.py` and `inpaint_model.py` aren't dependent on Apple's CDN
   / a third-party GitHub mirror.
2. **Promote FLUX monkey-patched VAE methods** to real subclass methods
   (`flux_pano_gen_pipeline.py`, `flux_pano_fill_pipeline.py`).
3. **Add `mypy` to CI** once the type annotations are more complete.
4. **CRDT vector clocks** — `merge_snapshots` currently uses a simple
   "b wins" LWW strategy; a production implementation should attach
   hybrid logical clocks to each object.
5. **`submodules/ml-sharp` checkout** — the submodule is registered but
   not checked out by default. Either init it in setup instructions or
   document the opt-in install path more prominently.

---

## 7. Files modified in this improvement pass

| File | Changes |
|------|---------|
| `src/auroch_syna/worldgen/utils/general_utils.py` | Removed `device` param from `map_image_to_pano`; fixed `depth_match` mutation; fixed `convert_rgbd2mesh_panorama` device default |
| `src/auroch_syna/worldgen/utils/splat_utils.py` | Added `SH_C0`; fixed `torch.cross dim=`; unified pixel-centre sampling; fixed `mask_splat` reshape; added `valid_mask_out` to `convert_rgbd_to_gs` |
| `src/auroch_syna/worldgen/utils/lora_utils.py` | Derived block count from checkpoint; matched zero-tensor dtype |
| `src/auroch_syna/worldgen/models/flux_pano_fill_pipeline.py` | Fixed `latent_image_ids` `NameError` on early-return path |
| `src/auroch_syna/worldgen/pano_gen.py` | Replaced `device="mps"` defaults with `resolve_device()` |
| `src/auroch_syna/worldgen/worldgen.py` | Removed `device=` arg from `map_image_to_pano` call |
| `.github/workflows/smoke-import.yml` | Added lint, unit-test, and improved smoke-import jobs |
| `tests/test_worldgen_utils.py` | **New** — 13 unit tests for pure-Python utilities |
| `.gitmodules` | Removed dead `pytorch3d` entry; changed `ml-sharp` SSH URL to HTTPS |
| `pyproject.toml` | Removed conflicting `[tool.setuptools.dynamic]` version attr |
