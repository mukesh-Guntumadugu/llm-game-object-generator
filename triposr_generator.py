import os
import sys
import subprocess
import torch
from PIL import Image

# Add TripoSR repo to path
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

def generate_triposr_mesh(image_path, output_dir, prefix):
    print("[*] Initializing TripoSR LRM Pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # TripoSR requires strict floating point precision for CPU inference
    if device == "cpu":
        torch.set_default_tensor_type(torch.FloatTensor)
        
    # Use the locally downloaded weights to bypass huggingface-hub conflicts
    model = TSR.from_pretrained(
        os.path.join(os.path.dirname(__file__), "TripoSR_weights"),
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)

    print("[*] Pre-processing 2D image for 3D reconstruction...")
    image = Image.open(image_path).convert("RGBA")
    
    # TripoSR expects a solid white/gray background. Blend alpha channel.
    import numpy as np
    image = remove_background(image, rembg_session=None)
    image = resize_foreground(image, 0.85)
    
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    
    print("[*] Inferring 3D Geometry (this requires intense CPU math)...")
    with torch.no_grad():
        scene_codes = model([image], device=device)
        
    print("[*] Extracting High-Resolution Marching Cubes Mesh...")
    # Extract the mesh with GPU-optimized parameters for maximum fidelity
    meshes = model.extract_mesh(scene_codes, True, resolution=256, threshold=25.0)
    
    # Save the raw TripoSR mesh with vertex colors
    raw_obj_path = os.path.join(output_dir, f"{prefix}_model.obj")
    meshes[0].export(raw_obj_path)
    
    print("[*] Invoking Headless Blender for UV Unwrapping & Baking...")
    albedo_path = os.path.join(output_dir, f"{prefix}_albedo.png")
    
    # Run Blender
    blender_script = os.path.join(os.path.dirname(__file__), "blender_bake.py")
    subprocess.run(["blender", "--background", "--python", blender_script, "--", raw_obj_path, albedo_path], check=True)
    
    print("[+] True 3D TripoSR Mesh and Albedo map successfully generated!")
    return raw_obj_path, albedo_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("output_dir")
    parser.add_argument("prefix")
    args = parser.parse_args()
    generate_triposr_mesh(args.image_path, args.output_dir, args.prefix)
