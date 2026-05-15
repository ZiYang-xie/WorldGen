# Auroch Syna (Syna) — Improvement Audit

_Last updated: 2026-05-15_

## Summary

Syna is the local fork of `WorldGen` — a Python ML pipeline that turns a text
prompt or image into a 360° panorama, lifts that panorama to depth + Gaussian
splats (or a triangle mesh), and serves the scene through a Viser web viewer.
The codebase is a research prototype: monolithic Python, CUDA-first, heavy
diffusers/transformers stack, no test suite, no typed interfaces between
stages, and no service boundaries.

This document is now intentionally **non-concise**. It records concrete,
file-grounded findings and a prioritized roadmap for turning the prototype
into something that can (a) run reliably on non-CUDA hardware (Apple
Silicon / MPS), (b) be embedded as a service inside the larger Auroch
runtime, and (c) be safely iterated on by more than one person.

---

## 1. Current state of the working tree

### 1.1 In-flight, uncommitted work

`git status` shows ten modified files and a thicket of `*.before-*` backup
files left by interactive edits:

```
M  README.md, demo.py, src/worldgen/worldgen.py,
   src/worldgen/pano_gen.py, src/worldgen/pano_inpaint.py,
   src/worldgen/pano_sharp.py, src/worldgen/utils/lora_utils.py,
   src/worldgen/models/flux_pano_gen_pipeline.py,
   src/worldgen/models/flux_pano_fill_pipeline.py
?? src/worldgen/model_client.py
?? demo.py.before-mps-fix.20260514-202023
?? src/worldgen/pano_gen.py.before-full-nunchaku-disable.20260514-201850
?? src/worldgen/pano_gen.py.before-nunchaku-fallback.20260514-201756
?? src/worldgen/utils/lora_utils.py.before-nunchaku-fallback.20260514-201555
?? src/worldgen/utils/lora_utils.py.broken.20260514-201640
D  LICENSE, assets/logo.png
```

Action items:

1. **Delete the `*.before-*` and `*.broken.*` backup files.** Use git for
   history; leaving them in the tree is noise and they will eventually get
   committed by accident. (`git clean -nx '*.before-*' '*.broken.*'` to
   preview.)
2. **Investigate the deletion of `LICENSE` and `assets/logo.png`.** The
   README still references both. Either restore them or update the README
   in the same commit.
3. **Land the in-flight MPS / Nunchaku fallback work as a small series of
   focused commits** rather than one mega-commit, so the diff is reviewable.

### 1.2 Broken re-export package

`src/auroch_syna/` exists as an empty directory (no `__init__.py`), yet
`demo.py:13` and the README's Python examples import from it:

```python
from auroch_syna import SplatFile, WorldGen
```

This means **`python demo.py` does not currently import cleanly from a
fresh checkout.** The previous REPORT.md claimed this re-export was added,
but only the directory was created — the actual module is missing.

Fix: add `src/auroch_syna/__init__.py` that re-exports the public surface
from `worldgen`:

```python
# src/auroch_syna/__init__.py
from worldgen.worldgen import WorldGen
from worldgen.utils.splat_utils import SplatFile

__all__ = ["WorldGen", "SplatFile"]
```

…and add `auroch_syna` to `[tool.setuptools.packages.find]` (it should be
picked up automatically since `where = ["src"]`, but verify with
`pip install -e . && python -c "import auroch_syna"`).

### 1.3 Package name vs. project name drift

- `pyproject.toml` still declares `name = "worldgen"` and
  `version = {attr = "worldgen.__version__"}`.
- The README is titled **WorldGen**.
- The repo on disk is **Syna**, the intended public name is
  **Auroch Syna**, and the new import path is **auroch_syna**.

Pick one of two paths and execute it end-to-end in a single PR:

- **Option A (lightweight):** keep the distribution as `worldgen`, keep
  `worldgen` as the canonical import, and treat `auroch_syna` purely as a
  thin alias shim. Document the alias relationship in the README.
- **Option B (full rebrand):** rename the package to `auroch_syna`, move
  `src/worldgen/` → `src/auroch_syna/`, update every import, update
  `pyproject.toml`, regenerate `worldgen.egg-info`, and have
  `worldgen` itself become the back-compat shim (with a `DeprecationWarning`).

Option A is much cheaper and is the recommended near-term move. Defer
Option B until the API surface is actually stable.

---

## 2. Architecture observations

### 2.1 Pipeline shape

`WorldGen.generate_world` (`src/worldgen/worldgen.py:134`) is a 3-stage
pipeline:

1. **Panorama generation** — `generate_pano` → FLUX.1-dev (`t2s`) or
   FLUX.1-Fill-dev (`i2s`) with a WorldGen LoRA.
2. **Depth estimation** — `pred_pano_depth` using DA-2 (recently swapped
   in for UniK3D, per `README.md:56`).
3. **Lift to scene** — either `convert_rgbd_to_gs` (Gaussian splats) or
   `convert_rgbd2mesh_panorama` (triangle mesh). Optional `use_sharp`
   path runs `predict_equirectangular` from ml-sharp.

Optional 4th stage: `inpaint_bg_splat` runs OneFormer segmentation +
LaMa inpainting to fill behind foreground objects.

This shape is fine. The problems are at the seams between stages, not in
the stages themselves.

### 2.2 The `ModelClient` is the right idea but is half-built

`src/worldgen/model_client.py` introduces a small wrapper that lazily
builds and caches each sub-model. This is the correct seam for later
moving model execution out-of-process. But today:

- It is **only used inside `WorldGen.__init__`** — every other call site
  (`pred_pano_depth`, `gen_pano_image`, `gen_pano_fill_image`,
  `predict_equirectangular`) still receives the raw pipeline object and
  calls into it directly. There is no abstraction over *inference*, only
  over *construction*.
- It hides imports inside methods to defer side effects, which is good,
  but the cache key for `build_pano_gen_model` (`f"pano_{mode}_{lora_path}"`)
  collides if two callers pass `lora_path=None` for different modes —
  wait, it doesn't, because `mode` is in the key. Fine. But document the
  contract: two `WorldGen` instances created with different `low_vram`
  values will silently share a cached model with the first instance's
  flag, because `low_vram` is on the `ModelClient`, not in the cache key.

Recommended evolution:

1. Define a `ModelHandle` protocol with `.infer(inputs) -> outputs` so
   that callers don't depend on the concrete pipeline class.
2. Add an `OutOfProcessModelClient` implementation that speaks the same
   protocol but proxies to a sidecar process via shared memory or
   gRPC/UDS. This is the prerequisite for embedding Syna inside a host
   app that owns its own event loop and GPU context.
3. Move device/precision/low-VRAM into a `RuntimeConfig` dataclass that
   is passed at construction time, instead of being scattered across
   `device=`, `low_vram=`, `torch_dtype=` arguments at every call.

### 2.3 Nunchaku fallback is brittle

`src/worldgen/pano_gen.py:8-19` wraps the `nunchaku` import in a broad
`except Exception:` and stubs out `NunchakuFluxTransformer2dModel`,
`get_precision`, and `compose_lora`. The fallback for `get_precision`
returns `"bf16"`, which is then **used in cache paths and model IDs**
elsewhere — make sure no code path tries to actually instantiate a
Nunchaku transformer on a platform where it failed to import (it doesn't
today, because the `low_vram and NunchakuFluxTransformer2dModel is not
None` guard at line 28 handles it — but the guard must stay).

The `pyproject.toml` dependency line pins Nunchaku to a Linux x86_64
wheel:

```
nunchaku @ https://github.com/mit-han-lab/nunchaku/releases/download/v0.2.0/nunchaku-0.2.0+torch2.7-cp311-cp311-linux_x86_64.whl
```

This makes `pip install .` **fail outright** on macOS / Apple Silicon.
Fix by making it an optional extra:

```toml
[project.optional-dependencies]
lowvram = [
  "nunchaku @ https://.../nunchaku-0.2.0+torch2.7-cp311-cp311-linux_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64'",
]
```

Then `pip install .[lowvram]` on Linux, plain `pip install .` everywhere
else. The runtime fallback in `pano_gen.py` already handles the missing
import gracefully.

### 2.4 Device handling is inconsistent

- `build_pano_gen_model` defaults `device="mps"` but
  `WorldGen.__init__` defaults `device='cuda'`. `demo.py` picks
  `cuda if available else cpu` (note: not `mps`).
- `pipe.enable_model_cpu_offload()` is only called when `device == "cuda"`,
  but the equivalent for MPS (manual `.to("mps")` + careful dtype
  selection) is not done. On MPS, `bfloat16` is not fully supported in
  many ops — `float16` is usually safer.
- `torch.Generator("cpu")` is hardcoded in `gen_pano_image` — that's
  actually correct for reproducibility across devices, but worth a
  comment so nobody "fixes" it.

Action: introduce a `resolve_device(preferred: str | None) -> torch.device`
helper that picks `cuda > mps > cpu` and a `default_dtype(device)` helper
that picks `bfloat16` on CUDA, `float16` on MPS, `float32` on CPU. Use
both consistently.

### 2.5 No test suite, no CI

There are zero tests in the repo. For an ML pipeline that takes minutes
per run and downloads multi-GB checkpoints, end-to-end tests are
impractical — but **unit tests for the pure-Python utilities are very
practical** and currently missing:

- `quaternion_slerp` in `demo.py:17` — pure numpy, trivially testable.
- `map_image_to_pano`, `resize_img`, `depth_match`,
  `convert_rgbd2mesh_panorama` in `utils/general_utils.py`.
- `mask_splat`, `merge_splats`, `convert_rgbd_to_gs` in
  `utils/splat_utils.py`.

A minimal `pytest` setup plus a GitHub Actions job that runs `ruff check`
+ `pytest tests/unit` on every PR would catch the most common regressions
(import errors, signature drift, numpy/torch dtype mistakes) without
needing a GPU runner.

### 2.6 Submodule hygiene

`git status` shows `submodules/DA-2` and `submodules/viser` as
modified-but-not-committed (`m` flag, not `M`), meaning the submodule
checkouts have drifted from the recorded SHAs. Decide whether the drift
is intentional (then bump the recorded SHA) or accidental (then
`git submodule update --init`). Leaving them drifted means every fresh
clone gets a different build than the developer.

---

## 3. Cross-platform / runtime concerns

The original audit framed Syna as needing a "native deterministic core
(Rust/C++, wgpu + ECS), ML services, and adapter layers." That is the
right long-term shape if Syna becomes a product, but it is **not the
right next step** — there is no native runtime to integrate with yet,
and the Python pipeline is still where 100% of the iteration happens.

A more pragmatic ladder:

| Stage | Goal | Effort |
|-------|------|--------|
| 0 (now) | Fix broken imports, make `pip install` work on macOS, land in-flight commits | days |
| 1 | `ModelHandle` protocol; out-of-process model server (single Python process, gRPC) | 1–2 weeks |
| 2 | Stable wire format for panorama + depth + splats (protobuf or CBOR) so non-Python clients can consume the output | 1 week |
| 3 | Adapter that lets a host app (Swift/Rust) call into the model server and get back a streaming splat | 2–3 weeks |
| 4 | Native renderer (wgpu) consuming the splat stream; Python only generates, native renders | months |

Stages 0–2 are unambiguously worth doing. Stage 3+ should wait until
there's a concrete host app asking for the API.

---

## 4. Concrete near-term punch list

In priority order, smallest-first:

1. **Restore `auroch_syna` re-export** so `demo.py` runs.
   Files: `src/auroch_syna/__init__.py` (new, ~5 lines).
2. **Delete `*.before-*` and `*.broken.*` backups** from the tree.
3. **Move Nunchaku to a platform-conditional optional extra** in
   `pyproject.toml` so non-Linux installs succeed.
4. **Add `resolve_device` / `default_dtype` helpers** and route all of
   `worldgen.py`, `pano_gen.py`, `pano_depth.py`, `pano_sharp.py`,
   `pano_inpaint.py` through them.
5. **Restore or replace `LICENSE` and `assets/logo.png`**, or update the
   README to match reality.
6. **Add minimal `pytest` + `ruff` CI** with unit tests for the pure
   utility functions listed in §2.5.
7. **Bump or reset submodule SHAs** for `DA-2` and `viser` so fresh
   clones are deterministic.
8. **Document the `ModelClient` contract** (caching semantics around
   `low_vram` / `device`) in its docstring.
9. **Introduce a `RuntimeConfig` dataclass** and migrate one call site
   to it as a pilot before propagating.
10. **Sketch the `ModelHandle` protocol** in a design doc before writing
    any out-of-process plumbing.

---

## 5. Files inspected (key)

- `demo.py` — Viser viewer + CLI entry point.
- `pyproject.toml` — distribution metadata, dependency pins.
- `README.md` — public-facing docs.
- `src/worldgen/__init__.py`, `src/worldgen/worldgen.py` —
  `WorldGen` orchestrator.
- `src/worldgen/model_client.py` — new construction-side wrapper.
- `src/worldgen/pano_gen.py` — FLUX pipelines + Nunchaku fallback.
- `src/worldgen/pano_depth.py`, `pano_seg.py`, `pano_inpaint.py`,
  `pano_sharp.py` — per-stage modules.
- `src/worldgen/models/flux_pano_gen_pipeline.py`,
  `flux_pano_fill_pipeline.py`, `inpaint_model.py` — diffusers
  pipeline subclasses.
- `src/worldgen/utils/lora_utils.py`, `general_utils.py`,
  `splat_utils.py` — math + IO helpers.
- `src/auroch_syna/` — currently empty; should re-export `WorldGen`,
  `SplatFile`.

## 6. Out of scope (intentionally not addressed here)

- Quality of the generated splats / mesh — that is a model and
  hyperparameter problem, not an engineering one.
- Replacing FLUX or DA-2 with newer checkpoints.
- A native Rust/wgpu renderer (premature; see §3).
- A CRDT-based scene-sync layer (premature; no second client exists yet).
