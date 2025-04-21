import os
import torch
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from .models.flux_pano_gen_pipeline import FluxPipeline
from .models.flux_pano_fill_pipeline import FluxFillPipeline


def build_pano_gen_model(lora_path=None, device="cuda"):
    if lora_path is None:
        lora_path = hf_hub_download(repo_id="LeoXie/WorldGen", filename=f"models--WorldGen-Flux-Lora/worldgen_text2scene.safetensors")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device=device)
    print(f"Loading LoRA weights from: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()
    return pipe

def build_pano_fill_model(lora_path=None, device="cuda"):
    if lora_path is None:
        lora_path = hf_hub_download(repo_id="LeoXie/WorldGen", filename=f"models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors")
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, device=device)
    print(f"Loading LoRA weights from: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()
    return pipe

def gen_pano_image(
        model,
        prompt="", 
        output_path=None, 
        seed=42, 
        guidance_scale=7.0, 
        num_inference_steps=50, 
        height=800, 
        width=1600, 
        blend_extend=6,
        prefix="A high quality 360 panorama photo of",
        suffix="HDR, RAW, 360 consistent, omnidirectional",
    ):
    """Generates a panorama image using FLUX.1-dev and a LoRA."""
    prompt = f"{prefix}, {prompt}, {suffix}"
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

def gen_pano_fill_image(
        model,
        image,
        mask,
        prompt="a scene",
        output_path=None,
        seed=42,
        guidance_scale=30.0,
        num_inference_steps=50,
        height=800,
        width=1600,
        blend_extend=6,
        prefix="A high quality 360 panorama photo of",
        suffix="HDR, RAW, 360 consistent, omnidirectional",
    ):
    image = image.resize((width, height))
    mask = mask.resize((width, height))
    generator = torch.Generator("cpu").manual_seed(seed)
    prompt = f"{prefix} {prompt} {suffix}"
    image = model(
        prompt,
        height=height,
        width=width,
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        blend_extend=blend_extend
    ).images[0]

    if output_path is not None:
        image.save(output_path)
        print(f"Panorama image saved to {output_path}")
        
    return image