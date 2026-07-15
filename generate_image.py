import os
import torch
from diffusers import StableDiffusionXLPipeline

def main():
    print("[*] Loading Stable Diffusion XL (SDXL) model in 16-bit (float16)...")
    
    # Load SDXL in FP16 precision
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Move the model to your GB10 GPU
    print("[*] Moving model to GPU (GB10)...")
    pipe.to("cuda")
    
    prompt = "2d game asset, pixel art style, sprite, single object, white background, a magical glowing fire sword"
    print(f"[*] Generating image for prompt: '{prompt}'")
    
    # Run inference (SDXL typical steps: 30)
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        width=512,
        height=512
    ).images[0]
    
    output_filename = "sprite.png"
    image.save(output_filename)
    print(f"[+] Image successfully generated and saved to: {output_filename}")

if __name__ == "__main__":
    main()
