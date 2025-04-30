import os
import torch
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from .models.flux_pano_gen_pipeline import FluxPipeline
from .models.flux_pano_fill_pipeline import FluxFillPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
from nunchaku.lora.flux.compose import compose_lora
from .utils.lora_utils import compose_lora_with_fixes, load_and_fix_lora


def build_pano_gen_model(lora_path=None, device="cuda", low_vram=True):
    """Build a panorama generation model with optional Nunchaku low VRAM support."""
    if lora_path is None:
        lora_path = hf_hub_download(repo_id="LeoXie/WorldGen", filename=f"models--WorldGen-Flux-Lora/worldgen_text2scene.safetensors")
    
    if low_vram:
        # Get precision and initialize Nunchaku transformer
        precision = get_precision()
        print(f"Using Nunchaku with {precision} precision")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/svdq-{precision}-flux.1-dev",
            offload=True
        )
        # Initialize pipeline with Nunchaku transformer
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load and fix LoRA weights
        print(f"Loading LoRA weights from: {lora_path}")
        state_dict, _ = load_and_fix_lora(lora_path)
        transformer.update_lora_params(state_dict)
    else:
        # Standard pipeline initialization
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load LoRA weights using standard diffusers method
        print(f"Loading LoRA weights from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    
    pipe.enable_model_cpu_offload() # Save VRAM
    pipe.enable_vae_tiling()
    return pipe

def build_pano_fill_model(lora_path=None, device="cuda", low_vram=True):
    """Build a panorama fill model with optional Nunchaku low VRAM support."""
    if lora_path is None:
        lora_path = hf_hub_download(repo_id="LeoXie/WorldGen", filename=f"models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors")
    
    if low_vram:
        # Get precision and initialize Nunchaku transformer
        precision = get_precision()
        print(f"Using Nunchaku with {precision} precision")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/svdq-{precision}-flux.1-fill-dev",
            offload=True
        )
        # Initialize pipeline with Nunchaku transformer
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load and fix LoRA weights
        print(f"Loading LoRA weights from: {lora_path}")
        state_dict, _ = load_and_fix_lora(lora_path)
        transformer.update_lora_params(state_dict)
    else:
        # Standard pipeline initialization
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            device=device
        )
        # Load LoRA weights using standard diffusers method
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