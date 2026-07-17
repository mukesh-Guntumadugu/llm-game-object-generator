import os
import sys
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from rembg import remove



# Adjust imports from local pipeline
from voxel_extruder import extrude_sprite_to_voxel_obj

def main():
    parser = argparse.ArgumentParser(description="Quickly generate a 3D Voxel Game Object from a text prompt.")
    parser.add_argument("prompt", type=str, help="What do you want to create? e.g., 'a glowing magic sword'")
    parser.add_argument("--style", type=str, default="voxels", help="Visual style (default: voxels)")
    args = parser.parse_args()

    # Clean the prompt for filenames
    import re
    import datetime
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', args.prompt.lower().replace(' ', '_')).strip('_')[:30]
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
    out_dir = os.path.join("generated_assets", "quick_generations", f"{safe_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Formatting the prompt
    full_prompt = f"3d voxel art rendering, single game object, isometric blocky style, stacked 3d cubes, isolated on solid white background, {args.prompt}"
    negative_prompt = "blurry, photo, realistic, smooth curves, flat 2d, gradient background, extra objects, text, watermark"
    
    print(f"\n==========================================")
    print(f"[*] Task: Generating '{args.prompt}'")
    print(f"[*] Style: {args.style}")
    print(f"[*] Output Directory: {out_dir}")
    print(f"==========================================\n")

    # 2. Loading SDXL model
    print("[*] Loading SDXL AI Model into GPU...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    
    print("[*] Generating 2D Base Image (this will take a few seconds)...")
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        width=512,
        height=512
    ).images[0]
    
    # Free up memory
    del pipe
    torch.cuda.empty_cache()

    # 3. Background Removal & Processing
    print("[*] Removing background from image...")
    image_nobg = remove(image)
    
    # Alpha threshold clamp for crisp edges
    r, g, b, a = image_nobg.split()
    a = a.point(lambda p: 255 if p > 100 else 0)
    final_sprite = Image.merge("RGBA", (r, g, b, a))
    
    sprite_path = os.path.join(out_dir, f"{safe_name}_sprite.png")
    final_sprite.save(sprite_path)
    print(f"[+] 2D Sprite saved at: {sprite_path}")

    # 4. 3D Voxel Extrusion (2.5D DPT Pipeline)
    print("[*] Converting to 3D Voxel Object using Adaptive Clarity...")
    obj_path = os.path.join(out_dir, f"{safe_name}_model.obj")
    
    # Call the extruder logic that we just upgraded
    extrude_sprite_to_voxel_obj(sprite_path, obj_path, style=args.style)
    
    print(f"\n[SUCCESS] Your beautiful object is ready! Check the '{out_dir}' folder!")

if __name__ == "__main__":
    main()
