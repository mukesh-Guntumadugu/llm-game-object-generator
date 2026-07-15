import os
import sys
from voxel_extruder import voxelize_mesh

if not os.path.exists("test_model.obj"):
    print("test_model.obj not found, skipping voxelize_mesh test.")
    sys.exit(0)

out_obj = "test_output/test_voxelized.obj"
os.makedirs("test_output", exist_ok=True)

print("Testing voxelize_mesh on test_model.obj...")
success = voxelize_mesh("test_model.obj", out_obj, grid_resolution=48)
print(f"Success: {success}")
