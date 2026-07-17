import os
from voxel_extruder import extrude_sprite_to_voxel_obj

test_images = [
    "greenscreen_nobg.png",
    "sprite.png"
]

os.makedirs("test_output", exist_ok=True)

for img in test_images:
    if os.path.exists(img):
        name = os.path.splitext(img)[0]
        out_path = f"test_output/{name}_voxel.obj"
        print(f"\n[*] Generating voxel object for {img}...")
        extrude_sprite_to_voxel_obj(img, out_path, target_size=128, style="voxels")
        print(f"[+] Saved to {out_path}")
    else:
        print(f"[!] {img} not found, skipping.")
