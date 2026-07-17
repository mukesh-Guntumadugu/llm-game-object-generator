import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
import trimesh
import shutil
import os

def generate_algorithmic_mesh(image_path, output_dir, prefix):
    """
    Converts a 2D sprite into a rounded 3D mesh using distance transforms and marching cubes.
    Requires no neural networks or C++ extensions.
    """
    print(f"[*] Loading 2D Image for Algorithmic 3D Extrusion: {image_path}")
    
    img = Image.open(image_path).convert("RGBA")
    
    # Resize to a standard resolution for processing
    PROCESS_SIZE = 128
    img = img.resize((PROCESS_SIZE, PROCESS_SIZE), Image.Resampling.LANCZOS)
    
    # Extract Alpha mask and RGBA color data
    img_data = np.array(img)
    alpha = img_data[:, :, 3]
    mask = alpha > 128
    
    if not np.any(mask):
        print("[!] Image is entirely transparent. Bailing out.")
        return None, None
    
    # Save the original image as the Albedo Texture NOW (before any processing)
    albedo_path = os.path.join(output_dir, f"{prefix}_albedo.png")
    shutil.copy2(image_path, albedo_path)
        
    print("[*] Computing Distance Transform for 3D Volume estimation...")
    # Distance transform: gives the distance from the edge for every pixel inside the mask
    distance = distance_transform_edt(mask)
    
    # Normalize distance to [0, 1]
    max_dist = np.max(distance)
    if max_dist > 0:
        normalized_depth = distance / max_dist
    else:
        normalized_depth = distance
        
    # Build a 3D grid that is CUBIC (same resolution in all dimensions)
    # This ensures the object looks round, not flat
    DEPTH_LAYERS = PROCESS_SIZE  # Same depth as width/height for round shapes
    CENTER_Z = DEPTH_LAYERS // 2
    MAX_RADIUS = int(DEPTH_LAYERS * 0.48)  # Max radius in Z direction
    
    print("[*] Building 3D Voxel Grid (spherical inflation)...")
    volume = np.zeros((PROCESS_SIZE, PROCESS_SIZE, DEPTH_LAYERS), dtype=np.float32)
    
    # For each pixel in the mask, we inflate it into a SPHERE in 3D
    # instead of a flat disc extrusion. This creates round apple/ball shapes.
    for y in range(PROCESS_SIZE):
        for x in range(PROCESS_SIZE):
            if mask[y, x]:
                # The depth radius is proportional to the distance from edge
                # => pixels at the center of a circle inflate into a sphere
                radius = normalized_depth[y, x] * MAX_RADIUS
                
                z_start = max(0, int(CENTER_Z - radius))
                z_end = min(DEPTH_LAYERS, int(CENTER_Z + radius) + 1)
                
                for z in range(z_start, z_end):
                    # Use spherical distance to smoothly taper the volume
                    dz = (z - CENTER_Z) / max(radius, 1e-5)
                    d2d = normalized_depth[y, x]
                    # Fill the voxel based on smooth spherical profile
                    if abs(dz) <= 1.0:
                        volume[y, x, z] = 1.0
    
    print("[*] Running Algorithmic Marching Cubes...")
    verts, faces, normals, _ = marching_cubes(volume, level=0.5)
    
    # Normalize vertices to [-1, 1] range preserving aspect ratio
    new_verts = np.zeros_like(verts)
    new_verts[:, 0] = (verts[:, 1] / PROCESS_SIZE) * 2.0 - 1.0       # X
    new_verts[:, 1] = ((PROCESS_SIZE - verts[:, 0]) / PROCESS_SIZE) * 2.0 - 1.0  # Y (flip Y axis)
    new_verts[:, 2] = (verts[:, 2] / DEPTH_LAYERS) * 2.0 - 1.0       # Z
    
    print("[*] Applying Taubin Smoothing to remove voxel staircase artifacts...")
    mesh = trimesh.Trimesh(vertices=new_verts, faces=faces)
    trimesh.smoothing.filter_taubin(mesh, iterations=20)
    smooth_verts = mesh.vertices
    
    print("[*] Generating Spherical UV Map (front-projection)...")
    # FIXED UV MAPPING: Use front-facing planar projection (XY plane)
    # This maps each vertex's X and Y position in screen space to UV coordinates.
    # This way the 2D image wraps onto the front-facing surface correctly.
    uvs = np.zeros((len(smooth_verts), 2))
    
    # Normalize X and Y to [0, 1] for UV
    x_min, x_max = smooth_verts[:, 0].min(), smooth_verts[:, 0].max()
    y_min, y_max = smooth_verts[:, 1].min(), smooth_verts[:, 1].max()
    
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    uvs[:, 0] = (smooth_verts[:, 0] - x_min) / x_range          # U from X
    uvs[:, 1] = 1.0 - (smooth_verts[:, 1] - y_min) / y_range    # V from Y (flip V)
    
    print(f"[*] Exporting final OBJ geometry: {prefix}_model.obj")
    obj_path = os.path.join(output_dir, f"{prefix}_model.obj")
    mtl_name = f"{prefix}_model.mtl"
    
    with open(obj_path, 'w') as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"o {prefix}\n")
        
        for v in smooth_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            
        f.write("usemtl BakeMat\n")
        f.write("s 1\n")
        for face in mesh.faces:
            v1, v2, v3 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
    
    # Write MTL file
    mtl_path = os.path.join(output_dir, mtl_name)
    with open(mtl_path, 'w') as f:
        f.write("newmtl BakeMat\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write(f"map_Kd {os.path.basename(albedo_path)}\n")
            
    print(f"[+] Algorithmic 3D Generation Complete! ({len(smooth_verts)} vertices, {len(mesh.faces)} faces)")
    return obj_path, albedo_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        generate_algorithmic_mesh(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python algorithmic_3d.py <image_path> <output_dir> <prefix>")
