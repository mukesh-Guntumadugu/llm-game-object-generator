import os
import torch
from diffusers import FluxPipeline

def main():
    print("[*] Loading FLUX.1-schnell model in 16-bit (bfloat16)...")
    
    # Load the FLUX.1-schnell pipeline using bfloat16 (16-bit)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    )
    
    # Move the model to CUDA (your GB10 GPU)
    print("[*] Moving model to GPU (GB10)...")
    pipe.to("cuda")
    
    prompt = "2d game asset, pixel art style, sprite, single object, white background, a magical glowing fire sword"
    print(f"[*] Generating image for prompt: '{prompt}'")
    
    # FLUX.1-schnell is designed to generate high-quality images in just 4 steps.
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512
    ).images[0]
    
    output_filename = "sprite.png"
    image.save(output_filename)
    print(f"[+] Image successfully generated and saved to: {output_filename}")

if __name__ == "__main__":
    main()
