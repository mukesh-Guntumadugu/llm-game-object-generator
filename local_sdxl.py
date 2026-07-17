"""
local_sdxl.py
Runs SDXL locally on the GPU to generate high-quality 2D pixel art game sprites.
Downloads the model once and caches it for reuse.
"""

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# Global pipeline cache so we only load the model once per session
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("[SDXL] Loading SDXL pipeline on GPU (first time may take a minute)...")
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    _pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    
    # Use DPM++ 2M Karras for fast high-quality generation
    _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        _pipeline.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    
    _pipeline = _pipeline.to("cuda")
    _pipeline.enable_vae_slicing()  # Save VRAM
    
    print("[SDXL] Pipeline ready!")
    return _pipeline


def generate_sprite(prompt: str, output_path: str, style: str = "pixel-art") -> str:
    """
    Generates a 2D game sprite using local SDXL.
    Returns the path to the saved image.
    """
    
    # Build style-specific prompt prefix and negative prompt
    if style == "pixel-art":
        full_prompt = (
            f"high quality 2D game asset, digital painting, RPG inventory icon, "
            f"detailed texture, studio lighting, isolated on pure white background, "
            f"professional game art, Diablo style, World of Warcraft style, "
            f"{prompt}"
        )
        negative = (
            "pixel art, blurry, bad anatomy, watermark, text, background, "
            "low quality, extra objects, multiple items, frame, border"
        )
    elif style == "vector-hand-drawn":
        full_prompt = (
            f"high quality 2D game asset, stylized hand-drawn digital painting, "
            f"clean illustration, isolated on white background, RPG item, {prompt}"
        )
        negative = "pixel art, photo, 3d render, blurry, watermark, background"
    else:
        full_prompt = (
            f"high quality 2D game asset, digital painting, RPG inventory icon, "
            f"detailed texture, studio lighting, white background, {prompt}"
        )
        negative = "pixel art, blurry, watermark, background, extra objects"

    print(f"[SDXL] Generating: {prompt}")
    
    pipe = get_pipeline()
    
    result = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        width=1024,
        height=1024,
        num_inference_steps=25,   # Fast but high quality
        guidance_scale=7.5,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    
    image = result.images[0]
    image.save(output_path)
    
    print(f"[SDXL] Saved sprite to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_sprite(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python local_sdxl.py 'A red apple' output.png")
