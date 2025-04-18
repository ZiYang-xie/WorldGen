import os
import torch
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from .models.flux_pano_pipeline import FluxPipeline

# Define constants for cache directories and URLs
WORLDGEN_CACHE_DIR = os.path.join(tempfile.gettempdir(), "worldgen_cache")
LORA_DEFAULT_PATH = os.path.join(WORLDGEN_CACHE_DIR, "pano_lora_720*1440_v1.safetensors")
CKPT_URL = "https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720*1440_v1.safetensors"

def ensure_cache_dir():
    """Create the cache directory if it doesn't exist"""
    os.makedirs(WORLDGEN_CACHE_DIR, exist_ok=True)
    return WORLDGEN_CACHE_DIR

def get_lora_path():
    """Get the path to the LoRA file, downloading it if necessary"""
    ensure_cache_dir()
    
    # Check if the file already exists in the cache
    if not os.path.exists(LORA_DEFAULT_PATH):
        print(f"LoRA weights not found in cache. Downloading...")
        import subprocess
        result = subprocess.run(
            ["wget", "-q", "--show-progress", CKPT_URL, "-O", LORA_DEFAULT_PATH],
            check=True
        )
        if result.returncode == 0:
            print(f"LoRA weights downloaded successfully to: {LORA_DEFAULT_PATH}")
        else:
            print(f"Warning: wget returned non-zero exit code: {result.returncode}")
        
        lora_path = LORA_DEFAULT_PATH
    else:
        print(f"Found LoRA weights in cache: {LORA_DEFAULT_PATH}")
        lora_path = LORA_DEFAULT_PATH
        
    return lora_path

def build_pano_gen_model(device="cuda"):
    lora_path = get_lora_path()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device="cuda")
    print(f"Loading LoRA weights from: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()       # Required for panorama generation
    return pipe

def gen_pano_image(
        model,
        prompt, 
        output_path=None, 
        seed=42, 
        guidance_scale=7.0, 
        num_inference_steps=50, 
        height=720, 
        width=1440, 
        blend_extend=6,
        prefix="A high quality 360 panorama photo of"
    ):
    """Generates a panorama image using FLUX.1-dev and a LoRA."""
    prompt = f"{prefix}, {prompt}"
    generator = torch.Generator("cpu").manual_seed(seed)
    image = model(
        prompt,
        height=height,
        width=width,
        generator=generator,
        num_inference_steps=num_inference_steps,
        blend_extend=blend_extend,
        guidance_scale=guidance_scale
    ).images[0]
    
    if output_path is not None:
        image.save(output_path)
        print(f"Panorama image saved to {output_path}")
        
    return image
