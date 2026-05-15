# Auroch Syna (Syna) — Improvement Audit

_Last updated: 2026-05-15_

## Summary

Syna is the local fork of `WorldGen` — a Python ML pipeline that turns a
text prompt or image into a 360° panorama, lifts that panorama to depth +
Gaussian splats (or a triangle mesh), and serves the scene through a
Viser web viewer. The codebase is a research prototype: monolithic
Python, CUDA-first, heavy diffusers/transformers stack, no real test
suite, no typed interfaces between stages, and no service boundaries.

This document is non-concise on purpose. It records concrete,
file-grounded findings and a prioritized roadmap. Companion doc
`ARCHITECTURE.md` covers the higher-level migration plan; this file
focuses on what's actually wrong in the code today.

---

## 0. Tree state and rebrand status (orient first)

During the in-flight rebrand, sources were moved:

```
src/worldgen/                  →  src/auroch_syna/worldgen/
worldgen.egg-info/             →  src/auroch_syna.egg-info/
new shim                       →  src/auroch_syna/__init__.py
new architecture doc           →  ARCHITECTURE.md
new CI workflow                →  .github/workflows/smoke-import.yml
```

The two rebrand commits are landed (`1ca385c`, `259de37`). Outstanding:

- **README still half-says "WorldGen"** (only the H1 and one section
  heading were renamed; bullets at `README.md:23,26,28,...` still talk
  about WorldGen). Either complete the rename or revert the partial one
  — half-renamed docs are worse than either.
- **Five `*.before-*` / `*.broken.*` backup files survived the move**
  and now live under `src/auroch_syna/worldgen/...` plus `demo.py.before-mps-fix...`
  in repo root. Delete them.
- `.gitignore` does not match `*.before-*` or `*.broken.*`. Add those
  patterns so this class of clutter can't be committed by accident.
- `submodules/pytorch3d` is registered in `.gitmodules` but never
  checked out (`git submodule status` shows no entry for it at all,
  only DA-2 / viser / ml-sharp). README does `pip install
  git+https://github.com/facebookresearch/pytorch3d.git` directly, so
  the submodule entry is dead weight — drop it from `.gitmodules`.
- `submodules/ml-sharp` has a recorded SHA but is not checked out
  (status line starts with `-`). README tells users to
  `pip install -e submodules/ml-sharp`, which silently fails on a
  fresh clone. Either init the submodule in setup instructions or
  change the install line to a URL.
- `submodules/DA-2` and `submodules/viser` are checked out but drifted
  from their recorded SHAs (`m` flag in git status). Either commit the
  bumped SHAs or `git submodule update --init` to reset.

---

## 1. The `auroch_syna` shim has bugs of its own

`src/auroch_syna/__init__.py` cleverly injects the old `worldgen/`
subdirectory onto `__path__` so existing `from .pano_depth import ...`
style imports still resolve during migration. Issues:

1. **Mixed tabs and spaces.** Lines 19–20 use tabs; the rest of the
   file uses spaces. Python 3 will reject this if a future edit lands
   on a tab-vs-space inconsistency. Normalize to spaces.
2. **Dead fallback path** (`__init__.py:34-35`): the `except`
   branch tries `import_module("worldgen.utils.splat_utils")`, but
   `worldgen` is no longer a top-level package after the rebrand —
   it's `auroch_syna.worldgen`. So the fallback can only ever raise
   `ModuleNotFoundError`. Either delete it or fix the path to
   `auroch_syna.worldgen.utils.splat_utils`.
3. **Bare `except Exception:`** at line 34 will swallow real bugs
   (e.g., a broken `splat_utils` will mask itself). Narrow to
   `ImportError` at minimum.
4. **`SplatFile` is imported from `auroch_syna.utils.splat_utils`
   first**, but that path doesn't exist either — the file is at
   `auroch_syna.worldgen.utils.splat_utils`. So in practice the lazy
   loader always hits the broken fallback. **This means
   `from auroch_syna import SplatFile` is currently broken** —
   `demo.py:13` imports `SplatFile` from there. Verify by running
   `python -c "from auroch_syna import SplatFile"`.

Fix: collapse the lazy loader to a single, correct path:

```python
def __getattr__(name: str):
    if name == "WorldGen":
        from auroch_syna.worldgen.worldgen import WorldGen
        return WorldGen
    if name == "SplatFile":
        from auroch_syna.worldgen.utils.splat_utils import SplatFile
        return SplatFile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## 2. Real correctness bugs found in the code

### 2.1 `flux_pano_fill_pipeline.py:714` — `NameError` on the `latents` early-return path

```python
def prepare_latents(self, image, timestep, batch_size, num_channels_latents,
                    height, width, dtype, device, generator, latents=None):
    ...
    if latents is not None:
        return latents.to(device=device, dtype=dtype), latent_image_ids
    # latent_image_ids is computed below this line, never above
```

`latent_image_ids` is never bound before the `if latents is not None`
branch returns it, and it is not a parameter. Any caller that passes a
non-`None` `latents` will get `NameError`. Today no internal caller
does, which is why it hasn't blown up — but anyone composing the
pipeline (e.g. for img2img-style flows) will hit this. Fix: either
compute `latent_image_ids` before the guard, or remove the early-return
and let the normal path execute.

### 2.2 `lora_utils.py:90-100` — silent LoRA dropping on macOS

```python
def compose_lora_with_fixes(lora_paths):
    fixed_loras = [load_and_fix_lora(path) for path, weight in lora_paths]
    if _nunchaku_compose_lora is None:
        print("[WorldGen] nunchaku unavailable on macOS; using first fixed LoRA fallback.")
        if not fixed_loras:
            return {}
        first_state_dict, _weight = fixed_loras[0]
        return first_state_dict
    return _nunchaku_compose_lora(fixed_loras)
```

When Nunchaku isn't available, **all LoRAs after the first are
silently discarded and the first LoRA's weight is ignored**. The
caller has no way to know. At minimum this should `warnings.warn`
loudly and include the count/identities of the dropped LoRAs. Better:
implement an actual diffusers-native compose fallback so multi-LoRA
prompts behave the same on every platform.

### 2.3 `pano_depth.py:44-92` — `pred_pano_depth` and `pred_depth` are byte-identical

Both functions have the exact same body (`np.array → permute → autocast
→ model → squeeze → normalize → pano_unit_rays`). One of them is dead
code, or the intent was different (e.g. `pred_depth` shouldn't compute
`pano_unit_rays` for non-pano inputs) and was lost during a refactor.
Decide which it is. If they really should be identical, delete one and
have the other be an alias.

### 2.4 `pano_seg.py:11` — debug remnant

```python
torch.set_float32_matmul_precision(['high', 'highest'][0])
```

The `['high', 'highest'][0]` is a leftover from someone toggling
between two values. Replace with a plain string and decide
intentionally which precision you want for OneFormer inference.

### 2.5 Fragile float equality (two files)

```python
assert (H / W == 0.5), "Input image aspect ratio is not 2:1. Is it a panorama?"
```

Appears in **both** `pano_seg.py:34` and `pano_inpaint.py:22`. Exact
float equality on `H/W` will reject perfectly valid panoramas like
`(2049, 1024)` or `(2048, 1023)` that round into floats slightly off
0.5. Use `abs(H/W - 0.5) < 1e-3` or `H * 2 == W` directly.

### 2.6 `splat_utils.py:124-149` — `mask_splat` assumes unfiltered image order

`mask_splat` does `centers.reshape(H, W, 3)[valid_mask]`, treating the
splat array as if it were still in row-major image order. But
`convert_rgbd_to_gs` at lines 71-75 **filters out pixels where
`distance <= dis_threshold`** before producing the splat. As soon as
the filter drops any pixel, the array is no longer `H*W` long and the
reshape either crashes (`RuntimeError: shape '[H, W, 3]' is invalid`)
or, if `H*W` happens to still divide evenly, silently produces
garbage. It only "works" today because `dis_threshold=0.0` and DA-2
returns strictly positive distances. Fix options:

- Carry the original `valid_mask` on the `SplatFile` and intersect it
  with the caller's mask before reshape.
- Stop filtering in `convert_rgbd_to_gs` and use opacity/scale=0 for
  invalid pixels instead, so the shape is stable.

### 2.7 `splat_utils.py:93,97,99` — deprecated `torch.cross` calls

`torch.cross(up, valid_rays)` (and two more) omit the `dim=` argument.
Modern PyTorch (≥2.0) emits `UserWarning: Using torch.cross without
specifying the dim arg is deprecated`, and is slated to make this an
error. Always pass `dim=1` (or `dim=-1`) explicitly.

### 2.8 `general_utils.py:148` — device-mismatch trap in `map_image_to_pano`

```python
def map_image_to_pano(predictions, ..., device: torch.device = 'cuda'):
    rgb_src = predictions["rgb"].float()      # lives on predictions' device
    ...
    rays_pano = pano_unit_rays(map_h, map_w, device)  # uses parameter device
    ...
    rays_hole = rays_pano[hole_mask]          # cross-device indexing
```

Every other tensor in the function lives on `rgb_src.device`, but
`rays_pano` is built on the `device` *parameter* (default `'cuda'`).
A caller who passes an MPS or CPU input but doesn't override
`device=` gets a cross-device indexing crash at line 154 or 170.
Drop the `device` parameter and use `rgb_src.device` throughout, or
plumb `device` through every allocation.

### 2.9 `general_utils.py:184-188` — `depth_match` silently mutates its input

```python
def depth_match(init_pred, bg_pred, mask) -> dict:
    ...
    bg_pred["distance"] *= scale     # in-place
    return bg_pred                    # returns the same dict
```

The caller's `bg_pred` dict is modified in place. Calling
`depth_match` twice on the same dict double-scales the distances.
Either deep-copy at the top, return a new dict, or rename
`depth_match_inplace` so the mutation is explicit.

---

## 3. Portability / device-handling issues

The earlier `__init__` defaults and MPS handling have not been
unified. Concrete callsites still hardcoding `cuda`:

| File | Line | What | Fix |
|------|------|------|-----|
| `pano_depth.py` | 38 | `def build_depth_model(device: torch.device = 'cuda')` | Use `resolve_device(None)` helper |
| `pano_seg.py` | 8 | `def build_segment_model(device: torch.device = 'cuda')` | Same |
| `pano_inpaint.py` | 7 | `def build_inpaint_model(device: torch.device = 'cuda')` | Same |
| `models/inpaint_model.py` | 22 | `LaMa(device='cuda')` | Same |
| `worldgen.py` | 20 | `device: torch.device = 'cuda'` | Same |
| `pano_gen.py` | 23, 63 | `device="mps"` *(!)* default | Pick one default; document why |

Also, `pano_depth.py:50,75` does `with torch.autocast(model.device.type)`
which **raises on CPU** (`torch.autocast` only supports `'cuda'`,
`'cpu'` since recent torch — verify your minimum torch version — and
`'mps'`, `'xpu'`). Wrap in a guard that no-ops on unsupported devices.

`pano_depth.py:96` references `data/background/timeless_desert.png` from
its `__main__` block; `data/` is in `.gitignore` so this never works
on any fresh checkout. Either ship a tiny test fixture or delete the
`__main__` block.

`pano_sharp.py` hardcodes an Apple CDN URL for the Sharp weights with
no fallback or checksum. If Apple rotates the URL, every fresh install
breaks silently. Mirror the weight on the WorldGen HuggingFace repo
and pin a sha256.

`pano_sharp.py:9-18` **imports `sharp.*` at module top level.** Any
code path that touches `pano_sharp` (e.g. `worldgen.py:51,91` does
`from .pano_sharp import predict_equirectangular` inside a
`use_sharp=True` branch — fine — but `model_client.py:70` does
`from .pano_sharp import build_sharp_model` which triggers the module
import) requires ml-sharp installed even when the feature is
disabled. Defer the `sharp.*` imports into `build_sharp_model` and
`predict_*` function bodies.

`pano_sharp.py:75,79-84` builds small tensors on CPU implicitly
(`torch.tensor([f_px / face_size]).float().to(device)`,
`torch.tensor([[...]], dtype=...).to(device)`), then ships them.
Construct directly on `device` with explicit `dtype` instead.

`pano_inpaint.py:36` writes `pano_inpainted_image.png` to the
**current working directory** unconditionally — should be an explicit
argument, defaulting to `None` (no save).

`models/inpaint_model.py:6-10` points `LAMA_MODEL_URL` at
`github.com/Sanster/models/releases/...` (a third-party
redistribution mirror) by default. If that mirror disappears, every
inpaint install silently breaks. Pin a primary + fallback, or
self-host on the WorldGen HF repo with a checksum.

---

## 4. Architectural smells (not bugs, but load-bearing tech debt)

### 4.1 `ModelClient` only wraps construction, not inference

`src/auroch_syna/worldgen/model_client.py` lazily builds and caches
each sub-model. That's the right seam, but every downstream call site
still receives the raw pipeline object and calls `pipe(...)` directly.
There is no abstraction over **inference**, which is the part that
would actually need to change for an out-of-process model server.

Define a `ModelHandle` protocol with `.infer(inputs) -> outputs` and
make `ModelClient` return handles, not raw pipelines. Then an
`OutOfProcessModelClient` can be a drop-in.

Also, the cache key for `build_pano_gen_model` includes `lora_path` and
`mode` but not `low_vram` or `device` — both of which live on the
`ModelClient` instance. So if you ever create two clients with
different settings, the second one wins silently because the
underlying cache is local to its own instance. That's fine today
(only one client per process) but should be documented in the
docstring before it surprises someone.

### 4.2 LoRA "fix" logic is suspicious

`lora_utils.py:75-85` walks `range(29)` for both `single_transformer_blocks`
and `transformer_blocks` and stuffs `torch.zeros(shape)` into every
missing entry. The `29` is FLUX.1-dev–specific magic. The fact that
this is needed at all suggests the LoRA was trained against a
checkpoint with a different block count, or that the diffusers loader
fails to materialize zero-rank blocks. Either way: (a) `29` should be
derived from the loaded model's actual block count, not hardcoded, and
(b) `torch.zeros(shape)` uses default dtype (float32) and default
device (CPU), which will require an implicit upcast on every block —
match the dtype of the existing LoRA tensors instead.

### 4.3 FLUX pipelines monkey-patch the VAE inside the inference loop

In `models/flux_pano_gen_pipeline.py` (and the fill variant), two
helper functions (`_decode`, `tiled_decode`) are defined locally
inside `__call__` and dynamically bound to `self.vae` with
`__get__()`. This is unmaintainable: undebuggable, untestable,
re-binds on every call. Promote them to real subclass methods on a
`PanoFluxVAE` mixin, or to standalone module-level functions that
take the VAE as a parameter.

The same files re-pack and re-unpack latents on every denoising step
inside the loop. Many of those reshape+permute ops can be hoisted
outside the loop and reused across steps.

### 4.4 No tests, no linter, weak CI

The new `.github/workflows/smoke-import.yml` is a useful first step
but it does **not run `pip install`** — it only imports the shim,
which uses `__getattr__` to defer all heavy imports. So the CI
currently catches syntax errors in `__init__.py` and nothing else.

Three immediate upgrades:

1. **Actually install the package** in CI (`pip install .` or
   `pip install -e .`) so import-side-effect errors and dependency
   resolution failures are caught. Skip `nunchaku` via the optional
   extra (see §3 of previous draft) so the install works on Linux
   runners without GPU.
2. **Add `ruff check`** with a minimal config (catch unused imports,
   undefined names — would have caught §2.1 above).
3. **Add a pytest job** with a `tests/unit/` directory containing
   tests for the pure-Python utilities (`quaternion_slerp`,
   `pano_unit_rays`, `resize_img`, `depth_match`, `mask_splat`,
   `merge_splats`). These don't need a GPU.

### 4.5 Diagnostics use `print()` everywhere

`pano_sharp.py:160,162,185`, `pano_gen.py:31,44,55,95,131`,
`pano_seg.py:37`, `pano_inpaint.py:26`, and `worldgen.py:36` all use
bare `print()` for diagnostic output. Callers (including the Viser
server, future CLI, future model service) have no way to silence or
redirect this. Standardize on a single `logging.getLogger(__name__)`
per module and let callers configure verbosity.

### 4.6 `worldgen/__init__.py` is missing `SplatFile`

`src/auroch_syna/worldgen/__init__.py` only exposes `WorldGen`. The
shim's lazy loader for `SplatFile` (§1.4) needs a stable path —
add `from .utils.splat_utils import SplatFile` here so both the shim
and any user doing `from auroch_syna.worldgen import SplatFile`
work.

### 4.7 Magic constants in `splat_utils.py`

- Line 27: `(self.rgbs - 0.5) / 0.28209479177387814` — this is
  `1 / (2 * sqrt(pi))`, the SH degree-0 normalization. Name it
  `SH_C0 = 0.28209479177387814` at module top with a comment.
- Line 79: `theta = torch.linspace(0, torch.pi, H, device=device)` —
  uses pixel-edge sampling, but `pano_unit_rays` in
  `general_utils.py:84-95` uses pixel-center sampling
  (`(arange + 0.5) / H`). The two are half a pixel apart for the
  same panorama. This produces a small but systematic bias in the
  per-pixel covariance scale. Either unify to pixel-center sampling
  or document the discrepancy.

### 4.8 `ARCHITECTURE.md` overlaps with this file

`ARCHITECTURE.md` (newly committed) restates the "split runtime / ML
service / adapters" thesis and lists immediate action items that
duplicate some of §0 here. Decide which doc owns which scope:

- **REPORT.md** (this file): findings, bugs, near-term punch list.
- **ARCHITECTURE.md**: target architecture, migration plan, design
  primitives.

…and have each link to the other. Right now both try to enumerate
"immediate action items" and they will drift.

---

## 5. Concrete near-term punch list

In priority order, smallest-first:

1. **Fix the broken `SplatFile` re-export** (`auroch_syna/__init__.py`)
   so `from auroch_syna import SplatFile` works, **and** export
   `SplatFile` from `auroch_syna/worldgen/__init__.py`. (§1.4, §4.6)
2. **Fix `flux_pano_fill_pipeline.py:714` NameError** on the
   `latents is not None` early-return. (§2.1)
3. **Fix the silent multi-LoRA drop** in `compose_lora_with_fixes`
   (or at minimum, `warnings.warn` instead of `print`). (§2.2)
4. **Fix deprecated `torch.cross` calls** in `splat_utils.py:93,97,99`
   by passing `dim=1`. (§2.7)
5. **Delete `pred_depth` or differentiate it from `pred_pano_depth`**.
   (§2.3)
6. **Remove the `['high', 'highest'][0]` debug remnant.** (§2.4)
7. **Fix `H/W == 0.5` float-equality check** in `pano_seg.py:34` and
   `pano_inpaint.py:22`. (§2.5)
8. **Fix `depth_match` in-place mutation** or rename it. (§2.9)
9. **Fix `map_image_to_pano` device-mismatch trap** (drop the
   `device` parameter or thread it through every allocation). (§2.8)
10. **Audit `mask_splat`'s reshape assumption** against
    `convert_rgbd_to_gs`'s filter. (§2.6)
11. **Delete the 5 `*.before-*` / `*.broken.*` backup files** and add
    those patterns to `.gitignore`.
12. **Complete or revert the README rename.**
13. **Drop `submodules/pytorch3d` from `.gitmodules`** (it's installed
    via pip URL anyway), and decide whether `submodules/ml-sharp`
    should be checked out by default or scrubbed from `.gitmodules`
    in favor of an opt-in install path.
14. **Defer `sharp.*` imports** inside `pano_sharp.py` so the module
    is importable without ml-sharp installed. (§3)
15. **Make the CI workflow actually install the package** and run
    `ruff check`. (§4.4)
16. **Add `resolve_device` / `default_dtype` helpers** and route every
    `device='cuda'` default through them. (§3)
17. **Mirror Sharp + LaMa weights** on HuggingFace with pinned
    sha256s so `pano_sharp.py` and `inpaint_model.py` aren't dependent
    on Apple's CDN / a third-party GitHub mirror. (§3)
18. **Switch all `print()` diagnostics to `logging`.** (§4.5)
19. **Promote FLUX monkey-patched VAE methods** to real subclass
    methods (`flux_pano_gen_pipeline.py:785,807,871-872`). (§4.3)
20. **Name the `SH_C0` constant** in `splat_utils.py:27` and unify
    pixel-center sampling between `splat_utils.py:79` and
    `general_utils.py:84-95`. (§4.7)
21. **Sketch a `ModelHandle` protocol** before writing out-of-process
    plumbing. (§4.1)
22. **Add `tests/unit/`** with tests for the pure utilities. (§4.4)

---

## 6. Files inspected (key)

- `demo.py` — Viser viewer + CLI entry point.
- `pyproject.toml` — distribution metadata, dependency pins.
- `README.md` — public-facing docs.
- `ARCHITECTURE.md` — sibling doc.
- `.github/workflows/smoke-import.yml` — CI.
- `src/auroch_syna/__init__.py` — re-export shim.
- `src/auroch_syna/worldgen/__init__.py`, `worldgen.py` —
  `WorldGen` orchestrator.
- `src/auroch_syna/worldgen/model_client.py` — construction wrapper.
- `src/auroch_syna/worldgen/pano_gen.py` — FLUX pipelines + Nunchaku fallback.
- `src/auroch_syna/worldgen/pano_depth.py` — DA-2 depth (duplicate functions).
- `src/auroch_syna/worldgen/pano_seg.py` — OneFormer segmentation.
- `src/auroch_syna/worldgen/pano_inpaint.py` — LaMa inpainting.
- `src/auroch_syna/worldgen/pano_sharp.py` — Apple ml-sharp adapter.
- `src/auroch_syna/worldgen/models/flux_pano_gen_pipeline.py`,
  `flux_pano_fill_pipeline.py`, `inpaint_model.py`.
- `src/auroch_syna/worldgen/utils/lora_utils.py` — LoRA load/compose.
- `src/auroch_syna/worldgen/utils/general_utils.py`,
  `splat_utils.py` — math + IO helpers.

## 7. Out of scope (intentionally not addressed here)

- Quality of the generated splats / mesh — that's a model and
  hyperparameter problem, not an engineering one.
- Replacing FLUX or DA-2 with newer checkpoints.
- A native Rust/wgpu renderer (premature; see ARCHITECTURE.md).
- A CRDT-based scene-sync layer (premature; no second client exists yet).
