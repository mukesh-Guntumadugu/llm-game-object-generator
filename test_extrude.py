import os
from voxel_extruder import extrude_sprite_to_voxel_obj

sprite = "greenscreen_nobg.png"
out_obj = "test_output/test.obj"

os.makedirs("test_output", exist_ok=True)

print(f"Testing extrusion on {sprite}...")
success = extrude_sprite_to_voxel_obj(sprite, out_obj, target_size=128, style="voxels")
print(f"Success: {success}")
