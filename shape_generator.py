import os
import subprocess
import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
from PIL import Image

def generate_shape_mesh(image_path, output_dir, prefix):
    """
    Generates a 3D mesh using OpenAI's Shap-E (Image-to-3D) diffusion model.
    """
    print("[*] Initializing OpenAI Shap-E Image-to-3D Pipeline...")
    
    # Load the PyTorch pipeline.
    # We use float16 since the user's GB10 GPU easily supports it natively.
    ckpt_id = "openai/shap-e-img2img"
    pipe = ShapEImg2ImgPipeline.from_pretrained(ckpt_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    print(f"[*] Processing 2D Reference Image: {image_path}")
    # Shap-E expects a standard PIL RGB image.
    image = Image.open(image_path).convert("RGB")
    
    print("[*] Inferring 3D Geometry via Neural Diffusion (Shap-E)...")
    # Generate the 3D mesh representation (NeRF/Marching Cubes implicitly)
    generator = torch.Generator(device="cuda").manual_seed(42)
    # 64 inference steps yields good quality for Shap-E
    images = pipe(
        image,
        generator=generator,
        guidance_scale=3.0,
        num_inference_steps=64,
        output_type="mesh"
    ).images
    
    # Export the raw mesh to OBJ
    raw_obj_path = os.path.join(output_dir, f"{prefix}_model_raw.obj")
    print(f"[*] Extracting Mesh to OBJ: {raw_obj_path}")
    export_to_obj(images[0], raw_obj_path)
    
    print("[*] Invoking Headless Blender for UV Unwrapping & Color Baking...")
    # Shap-E encodes color in the vertices. We must bake it into an Albedo texture
    # so standard game engines and our PBR pipeline can read it perfectly.
    albedo_path = os.path.join(output_dir, f"{prefix}_albedo.png")
    blender_script = os.path.join(os.path.dirname(__file__), "blender_bake.py")
    
    # Run the same robust blender script we used for TripoSR
    subprocess.run(["blender", "--background", "--python", blender_script, "--", raw_obj_path, albedo_path], check=True)
    
    # After baking, the blender script overwrites the OBJ with UVs mapped to the texture,
    # so we can use the same base name.
    final_obj_path = os.path.join(output_dir, f"{prefix}_model.obj")
    if os.path.exists(raw_obj_path) and not os.path.exists(final_obj_path):
        os.rename(raw_obj_path, final_obj_path)
    
    print("[+] Shap-E 3D Mesh and Albedo map successfully generated!")
    return final_obj_path, albedo_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        generate_shape_mesh(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python shape_generator.py <image_path> <output_dir> <prefix>")
