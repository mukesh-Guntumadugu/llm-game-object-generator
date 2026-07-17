import sys
import os
import torch
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))
from tsr.system import TSR
from tsr.utils import resize_foreground

image_path = "generated_assets/quick_generations/a_glowing_magic_sword/a_glowing_magic_sword_sprite.png"
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

print("Loading TSR...")
model = TSR.from_pretrained("TripoSR_weights", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(131072)
model.to("cuda:0")

image = Image.open(image_path).convert("RGBA")
# DO NOT call remove_background. The image already has an alpha channel!
import numpy as np
image = resize_foreground(image, 0.85)

image = np.array(image).astype(np.float32) / 255.0
image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
image = Image.fromarray((image * 255.0).astype(np.uint8))

print("Inferring...")
with torch.no_grad():
    scene_codes = model([image], device="cuda:0")
meshes = model.extract_mesh(scene_codes, True)
meshes[0].export(f"{output_dir}/fixed_triposr.obj")
print("Done.")
