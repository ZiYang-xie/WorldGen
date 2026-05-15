"""Auroch Syna extensions to HuggingFace diffusers pipelines.

These are adapted versions of upstream diffusers pipelines with panoramic
blending support. Attribution headers are preserved inside each module.

Exports:
    FluxPipeline     — FLUX.1-dev with panoramic blend (text-to-scene)
    FluxFillPipeline — FLUX.1-Fill-dev with panoramic blend (image-to-scene)
"""
from .flux_pano_gen_pipeline import FluxPipeline
from .flux_pano_fill_pipeline import FluxFillPipeline

__all__ = ["FluxPipeline", "FluxFillPipeline"]
