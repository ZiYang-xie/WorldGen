**Auroch Syna — Short Architecture & Migration Plan**

Purpose
- Capture prioritized, actionable items from the audit to evolve this repo into a cross‑platform, context‑aware world‑building environment.

Goals
- Separate deterministic runtime from ML service layer.
- Define semantic primitives for state and handoff.
- Make on‑device inference practical (CoreML/ONNX/TorchScript conversions).
- Add developer ergonomics (CLI, demo harness, packaging).

High Level Architecture
- Core: small, deterministic engine (Rust/C++), exposes an IPC/FFI boundary.
- ML Layer: model runners (local or remote) exposing a `ModelClient` API (gRPC/HTTP/UNIX socket).
- Adapters: bridge internal scene primitives to ML outputs (pano → splats/mesh).
- Orchestration: state sync, snapshot, diff & handoff using CRDTs or versioned snapshots.

Semantic Primitives
- Affordance<T> — typed, minimal state units (eg. `Affordance<Splat>`, `Affordance<Mesh>`).
- Snapshot / Delta — immutable snapshots + compact diffs for efficient sync/handoff.
- Intent / Command — serializable commands for interactive tools (Winnie/CLI).

Performance & Local Edge
- Convert heavy models to platform optimized formats (CoreML for iOS, TorchScript/ONNX for CPU/GPU).
- Add model warm/cold policy in `ModelClient` (lazy load, keep‑alive, offload).
- Add tiling & streaming for large pano decoding in `Flux` pipelines.

Service Integration
- Define minimal `ModelClient` interface (already present in `src/auroch_syna/worldgen/model_client.py`).
- Add pluggable transport: in‑process, unix socket, HTTP/gRPC.
- Provide CLI runner with config to target local model server vs in‑process.

Developer UX
- Provide lightweight smoke tests, CI job to check imports and basic pipeline instantiation.
- Supply a `auroch_syna` package that is importable while we migrate sources (already added).

Immediate Action Items (priority)
1. Complete rebrand and packaging: ensure `pyproject.toml`, egg‑info, README, and demos use `auroch_syna` (done for core files; still scan & fix leftovers).
2. Add architecture doc and roadmap (this file).
3. Add a smoke import test and CI job that runs: `python -c "from auroch_syna import WorldGen; print('OK')"`.
4. Convert critical model loads in `ModelClient` to support a transport interface (thin adapter pattern).
5. Add a `design/` folder for future proto files (gRPC schema), CRDT library picks, and ECS primitives.

Where to start in code
- `src/auroch_syna/worldgen/model_client.py` — boundary for ML loads.
- `src/auroch_syna/worldgen/worldgen.py` — high level API.
- `src/auroch_syna/worldgen/pano_*` — pipeline adapters.
- `src/auroch_syna/worldgen/utils/*` — utilities to keep; consider moving heavy image libs behind optional deps.

Next checkpoints
- (A) CI smoke import and lint (1–2 hours).  
- (B) Implement `ModelClient` transport adapters + example local runner (1–2 days).  
- (C) Design CRDT-based snapshot format + offline handoff prototype (2–4 days).

Appendix — quick checklist for packaging
- `pyproject.toml` name and dynamic version attr (updated).  
- `src/auroch_syna/__init__.py` shim (present).  
- Egg‑info & README updates (updated, but scan for remaining mentions).
