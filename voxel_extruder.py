import os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

def extrude_sprite_to_voxel_obj(sprite_path, output_obj_path, target_size=None, style="pixel-art"):
    """
    Extrudes a transparent 2D sprite into a volumetric 3D voxel model (.obj format)
    by mapping each non-transparent pixel to a column of 3D cubes.
    The Z-thickness (depth) of each pixel is determined using a Euclidean distance transform,
    creating rounded, organic volumetric shapes (thick in center, thin near edges).
    It optimizes the mesh by omitting all hidden/internal faces.
    """
    if not os.path.exists(sprite_path):
        print(f"[!] Sprite not found at {sprite_path}")
        return False
        
    try:
        # Load image and ensure RGBA
        img = Image.open(sprite_path).convert("RGBA")
        
        # Configure resolution and resampling method based on the style
        if target_size is None:
            target_size = 512
                
        resample_method = Image.Resampling.LANCZOS
        img.thumbnail((target_size, target_size), resample_method)
        width, height = img.size
        pixels = img.load()
        
        # Prepare OBJ and MTL file names
        base_dir = os.path.dirname(output_obj_path)
        base_name = os.path.splitext(os.path.basename(output_obj_path))[0]
        mtl_filename = f"{base_name}.mtl"
        output_mtl_path = os.path.join(base_dir, mtl_filename)
        
        # Copy the resized image to serve as the texture map
        texture_name = f"{base_name}_texture.png"
        texture_path = os.path.join(base_dir, texture_name)
        
        # Load high-res texture image from original sprite_path and save it
        try:
            original_img = Image.open(sprite_path).convert("RGBA")
            original_img = original_img.resize((512, 512), Image.Resampling.LANCZOS)
            original_img.save(texture_path)
        except Exception as e:
            print(f"[!] Could not save high-res texture: {e}, falling back to low-res.")
            img.save(texture_path)
        
        # Write MTL file
        with open(output_mtl_path, "w") as mtl_file:
            mtl_file.write(f"# Material for {base_name}\n")
            mtl_file.write(f"newmtl {base_name}_material\n")
            mtl_file.write("Ka 1.0 1.0 1.0\n")
            mtl_file.write("Kd 1.0 1.0 1.0\n")
            mtl_file.write("Ks 0.0 0.0 0.0\n")
            mtl_file.write("d 1.0\n")
            mtl_file.write("illum 1\n")
            mtl_file.write(f"map_Kd {texture_name}\n")
            
        print("[*] Generating AI Depth Map for 2.5D Extrusion...")
        from transformers import pipeline
        pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")
        depth_img = pipe(img.convert("RGB"))["depth"]
        depth_array = np.array(depth_img)
        
        # Normalize depth map to 0.0 - 1.0
        d_min = np.min(depth_array)
        d_max = np.max(depth_array)
        if d_max > d_min:
            depth_normalized = (depth_array - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(depth_array, dtype=float)
            
        # 1. Standard 2.5D Distance Transform (for edges and core mask)
        alpha = np.array(img)[:, :, 3]
        mask = alpha > 128
        distances = distance_transform_edt(mask)
        factor = 0.55 if style in ["realistic-high-poly", "voxels", "cel-shaded-stylized"] else 0.45
        max_thickness_cap = max(1, int(width * 0.22))
        scaled_distances = distances * factor
        
        # Base thickness from distance transform (keeps edges clean and centered)
        base_thickness = np.zeros_like(distances, dtype=float)
        base_thickness[mask] = np.minimum(scaled_distances[mask], max_thickness_cap)
        
        # Combine distance transform thickness with AI depth map offsets
        # The Depth Map determines the structural push/pull (Z-offset)
        # The Distance Transform determines the thickness of the voxel column
        
        # We will use the depth map to shift the center of the voxel column!
        depth_offset = (depth_normalized - 0.5) * (max_thickness_cap * 1.5)
        
        offset_int = np.round(depth_offset).astype(int)
        thickness_int = np.round(base_thickness).astype(int)
        thickness_int = np.maximum(thickness_int, 1) # Minimum 1 voxel thick
        
        # Compute the bounding box of Z to size the grid
        max_t = int(np.max(thickness_int)) if np.any(thickness_int) else 0
        max_abs_offset = int(np.max(np.abs(offset_int))) if np.any(offset_int) else 0
        depth_size = 2 * (max_t + max_abs_offset) + 1
        z_center = depth_size // 2
        grid_width = width
        
        grid = np.zeros((height, grid_width, depth_size), dtype=bool)
        for y in range(height):
            for x in range(grid_width):
                if mask[y, x]:
                    t = thickness_int[y, x]
                    off = offset_int[y, x]
                    # Create a column of voxels shifted by the AI depth
                    for z_idx in range(-t, t + 1):
                        grid[y, x, z_idx + off + z_center] = True
                        
        # CLEANUP: Connected Component Analysis
        # Ensure the 2.5D game object is properly connected, removing floating artifacts
        from scipy.ndimage import label
        labeled, num_features = label(grid)
        if num_features > 1:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # Ignore background
            largest = np.argmax(component_sizes)
            grid = (labeled == largest)
            print(f"[*] Cleanup: Kept largest component ({component_sizes[largest]} voxels), removed {num_features - 1} detached floating pieces")
                            
        # 3. Generate OBJ Mesh using Greedy Meshing for all 6 directions
        vertices_dict = {}
        vertices = []
        faces = []
        
        v_idx = 1
        hs = 0.5 # Half-size of a voxel cube
        
        def get_vertex_id(vx, vy, vz):
            nonlocal v_idx
            key = (round(vx, 4), round(vy, 4), round(vz, 4))
            if key not in vertices_dict:
                vertices_dict[key] = v_idx
                vertices.append(key)
                v_idx += 1
            return vertices_dict[key]
            
        uvs_dict = {}
        uvs = []
        uv_idx = 1
        
        def get_uv_id(u, v):
            nonlocal uv_idx
            # Clamp UVs
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            key = (round(u, 6), round(v, 6))
            if key not in uvs_dict:
                uvs_dict[key] = uv_idx
                uvs.append(key)
                uv_idx += 1
            return uvs_dict[key]
            
        def greedy_mesh_2d(mask_2d):
            h_dim, w_dim = mask_2d.shape
            visited = np.zeros((h_dim, w_dim), dtype=bool)
            quads = []
            for i in range(h_dim):
                for j in range(w_dim):
                    if mask_2d[i, j] and not visited[i, j]:
                        w_q = 1
                        while j + w_q < w_dim and mask_2d[i, j + w_q] and not visited[i, j + w_q]:
                            w_q += 1
                        h_q = 1
                        can_expand = True
                        while i + h_q < h_dim and can_expand:
                            for w_idx in range(w_q):
                                if not mask_2d[i + h_q, j + w_idx] or visited[i + h_q, j + w_idx]:
                                    can_expand = False
                                    break
                            if can_expand:
                                h_q += 1
                        visited[i:i+h_q, j:j+w_q] = True
                        quads.append((j, i, w_q, h_q))
            return quads

        # Generate voxel geometry using Greedy Meshing for all 6 directions
        
        # 1. Front Faces (+Z)
        for z in range(depth_size):
            if z == depth_size - 1:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z + 1]
            quads = greedy_mesh_2d(visible)
            z_val = z - depth_size / 2.0 + hs
            for (x, y, w, h) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                
                vt_bl = get_uv_id(x / grid_width, 1.0 - (y + h) / height)
                vt_br = get_uv_id((x + w) / grid_width, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                vt_tl = get_uv_id(x / grid_width, 1.0 - y / height)
                
                faces.append(f"{v_bl}/{vt_bl} {v_br}/{vt_br} {v_tr}/{vt_tr} {v_tl}/{vt_tl}")
                
        # 2. Back Faces (-Z)
        for z in range(depth_size):
            if z == 0:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z - 1]
            quads = greedy_mesh_2d(visible)
            z_val = z - depth_size / 2.0 - hs
            for (x, y, w, h) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                
                vt_bl = get_uv_id(x / grid_width, 1.0 - (y + h) / height)
                vt_br = get_uv_id((x + w) / grid_width, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                vt_tl = get_uv_id(x / grid_width, 1.0 - y / height)
                
                faces.append(f"{v_br}/{vt_br} {v_bl}/{vt_bl} {v_tl}/{vt_tl} {v_tr}/{vt_tr}")
                
        # 3. Right Faces (+X)
        for x in range(grid_width):
            if x == grid_width - 1:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x + 1, :]
            quads = greedy_mesh_2d(visible)
            x_val = x - grid_width / 2.0 + hs
            for (z, y, d, h) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                
                vt_bl = get_uv_id(z / depth_size, 1.0 - (y + h) / height)
                vt_br = get_uv_id((z + d) / depth_size, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((z + d) / depth_size, 1.0 - y / height)
                vt_tl = get_uv_id(z / depth_size, 1.0 - y / height)
                
                faces.append(f"{v_br}/{vt_br} {v_bl}/{vt_bl} {v_tl}/{vt_tl} {v_tr}/{vt_tr}")
                
        # 4. Left Faces (-X)
        for x in range(grid_width):
            if x == 0:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x - 1, :]
            quads = greedy_mesh_2d(visible)
            x_val = x - grid_width / 2.0 - hs
            for (z, y, d, h) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                
                vt_bl = get_uv_id(z / depth_size, 1.0 - (y + h) / height)
                vt_br = get_uv_id((z + d) / depth_size, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((z + d) / depth_size, 1.0 - y / height)
                vt_tl = get_uv_id(z / depth_size, 1.0 - y / height)
                
                faces.append(f"{v_bl}/{vt_bl} {v_br}/{vt_br} {v_tr}/{vt_tr} {v_tl}/{vt_tl}")
                
        # 5. Top Faces (+Y)
        for y in range(height):
            if y == 0:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y - 1, :, :]
            quads = greedy_mesh_2d(visible)
            y_val = height / 2.0 - y + hs
            for (z, x, d, w) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                
                vt_min = get_uv_id(x / grid_width, 1.0 - y / height)
                vt_max = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                
                faces.append(f"{v_bl}/{vt_min} {v_tl}/{vt_min} {v_tr}/{vt_max} {v_br}/{vt_max}")
                
        # 6. Bottom Faces (-Y)
        for y in range(height):
            if y == height - 1:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y + 1, :, :]
            quads = greedy_mesh_2d(visible)
            y_val = height / 2.0 - y - hs
            for (z, x, d, w) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                
                vt_min = get_uv_id(x / grid_width, 1.0 - (y + 1) / height)
                vt_max = get_uv_id((x + w) / grid_width, 1.0 - (y + 1) / height)
                
                faces.append(f"{v_bl}/{vt_min} {v_br}/{vt_max} {v_tr}/{vt_max} {v_tl}/{vt_min}")
                        
        with open(output_obj_path, "w") as obj_file:
            obj_file.write(f"# 3D Volumetric Voxel Object: {base_name}\n")
            obj_file.write(f"mtllib {mtl_filename}\n\n")
            
            # Write all vertices
            for vert in vertices:
                obj_file.write(f"v {vert[0]:.4f} {vert[1]:.4f} {vert[2]:.4f}\n")
            obj_file.write("\n")
            
            # Write UV coordinates
            for uvc in uvs:
                obj_file.write(f"vt {uvc[0]:.6f} {uvc[1]:.6f}\n")
            obj_file.write("\n")
            
            # Use material
            obj_file.write(f"usemtl {base_name}_material\n")
            
            # Write all faces
            for face in faces:
                obj_file.write(f"f {face}\n")
                
        print(f"[+] 3D Voxel Model successfully saved to {output_obj_path}")
        return True
    except Exception as e:
        print(f"[!] Error extruding voxel model: {e}")
        return False


def voxelize_mesh(mesh_obj_path, output_obj_path, sprite_texture_path=None, grid_resolution=48):
    """
    Voxelizes a true 3D mesh (e.g., from TripoSR) into a blocky voxel-style .obj file.
    
    This is the core of the Realistic-to-Voxel pipeline:
    1. Load the TripoSR mesh with trimesh
    2. Scale it to fit a grid_resolution^3 voxel grid
    3. Voxelize it into a boolean 3D grid
    4. Run the Greedy Mesher to produce an optimized .obj
    
    Args:
        mesh_obj_path: Path to the input .obj mesh (from TripoSR)
        output_obj_path: Path to write the voxelized .obj
        sprite_texture_path: Optional path to a texture image for the material
        grid_resolution: Size of the voxel grid (default 48 for classic blocky look)
    """
    if not os.path.exists(mesh_obj_path):
        print(f"[!] Mesh not found at {mesh_obj_path}")
        return False
    
    try:
        import trimesh
        
        print(f"[*] Loading TripoSR mesh from {mesh_obj_path}...")
        mesh = trimesh.load(mesh_obj_path, force='mesh')
        
        # Center the mesh at origin
        mesh.apply_translation(-mesh.centroid)
        
        # Scale to fit inside the voxel grid
        scale = (grid_resolution - 2) / np.max(mesh.extents)  # -2 for padding
        mesh.apply_scale(scale)
        
        # Translate so the minimum is at (0,0,0)
        mesh.apply_translation(-mesh.bounds[0])
        
        print(f"[*] Voxelizing mesh at {grid_resolution}x{grid_resolution}x{grid_resolution} resolution...")
        voxelized = mesh.voxelized(pitch=1.0)
        
        # CRITICAL: Fill the interior of the mesh shell to create SOLID voxels
        # TripoSR outputs hollow shells; without filling, we get thin noisy surfaces
        try:
            filled = voxelized.fill()
            grid = filled.matrix
            print(f"[*] Filled interior: {np.sum(voxelized.matrix)} → {np.sum(grid)} voxels")
        except Exception:
            grid = voxelized.matrix
        
        # The grid axes from trimesh are (X, Y, Z)
        # Our greedy mesher expects (height/Y, width/X, depth/Z)
        # Trimesh: axis 0 = X (left-right), axis 1 = Y (up-down), axis 2 = Z (front-back)
        # We need: axis 0 = Y (rows, top-down), axis 1 = X (columns, left-right), axis 2 = Z (depth)
        grid = np.transpose(grid, (1, 0, 2))
        # Flip Y so top of object is at index 0 (image convention)
        grid = np.flip(grid, axis=0)
        
        # CLEANUP 1: Connected Component Analysis — keep ONLY the largest blob
        # This strips all floating debris and scattered voxels
        from scipy.ndimage import label, binary_closing, generate_binary_structure
        labeled, num_features = label(grid)
        if num_features > 1:
            # Find the largest connected component
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # Ignore background
            largest = np.argmax(component_sizes)
            grid = (labeled == largest)
            print(f"[*] Cleanup: Kept largest component ({component_sizes[largest]} voxels), removed {num_features - 1} floating debris clusters")
        
        # CLEANUP 2: Morphological closing — fill small 1-voxel holes and smooth spikes
        struct = generate_binary_structure(3, 1)  # 6-connected
        grid = binary_closing(grid, structure=struct, iterations=1)
        
        # ADAPTIVE CLARITY (3D Quantization):
        # 1. Compute 3D distance transform to measure local thickness
        from scipy.ndimage import distance_transform_edt, maximum_filter
        distances = distance_transform_edt(grid)
        
        # 2. Propagate core thicknesses out to the surface
        local_thickness = maximum_filter(distances, size=5)
        
        # 3. Create block-quantized versions of the grid
        def block_quantize(mask, block_size):
            if block_size <= 1:
                return mask
            shape = mask.shape
            pad_r = [(0, (block_size - s % block_size) % block_size) for s in shape]
            padded = np.pad(mask, pad_r, mode='constant')
            
            new_shape = (padded.shape[0] // block_size, block_size,
                         padded.shape[1] // block_size, block_size,
                         padded.shape[2] // block_size, block_size)
            blocks = padded.reshape(new_shape)
            
            # If > 30% filled, consider the block solid
            block_filled = blocks.mean(axis=(1, 3, 5)) > 0.3
            
            quantized = np.repeat(np.repeat(np.repeat(block_filled, block_size, axis=0),
                                                    block_size, axis=1),
                                          block_size, axis=2)
            return quantized[:shape[0], :shape[1], :shape[2]]
            
        grid_q2 = block_quantize(grid, 2)
        grid_q3 = block_quantize(grid, 3)
        
        # 4. Blend based on local thickness:
        # Thin edges (thickness < 3) keep 1x1x1 high clarity
        # Mid regions (3 <= thickness < 6) use 2x2x2 blocks
        # Core regions (thickness >= 6) use 3x3x3 blocks
        final_grid = np.copy(grid)
        mask_q2 = (local_thickness >= 3) & (local_thickness < 6)
        mask_q3 = (local_thickness >= 6)
        
        final_grid[mask_q2] = grid_q2[mask_q2]
        final_grid[mask_q3] = grid_q3[mask_q3]
        
        grid = final_grid
        
        # Re-run Connected Component Analysis just in case block quantization disconnected thin bridges
        labeled, num_features = label(grid)
        if num_features > 1:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            largest = np.argmax(component_sizes)
            grid = (labeled == largest)
        
        height, grid_width, depth_size = grid.shape
        print(f"[*] Final voxel grid: {height}x{grid_width}x{depth_size} ({np.sum(grid)} filled voxels)")
        
        if np.sum(grid) == 0:
            print("[!] Voxelization produced an empty grid!")
            return False
        
        # Prepare OBJ and MTL file names
        base_dir = os.path.dirname(output_obj_path)
        base_name = os.path.splitext(os.path.basename(output_obj_path))[0]
        mtl_filename = f"{base_name}.mtl"
        output_mtl_path = os.path.join(base_dir, mtl_filename)
        
        # Set up texture
        texture_name = f"{base_name}_texture.png"
        texture_path = os.path.join(base_dir, texture_name)
        
        if sprite_texture_path and os.path.exists(sprite_texture_path):
            try:
                tex_img = Image.open(sprite_texture_path).convert("RGBA")
                tex_img = tex_img.resize((512, 512), Image.Resampling.LANCZOS)
                tex_img.save(texture_path)
            except Exception as e:
                print(f"[!] Could not copy texture: {e}")
        
        # Write MTL file
        with open(output_mtl_path, "w") as mtl_file:
            mtl_file.write(f"# Material for {base_name}\n")
            mtl_file.write(f"newmtl {base_name}_material\n")
            mtl_file.write("Ka 1.0 1.0 1.0\n")
            mtl_file.write("Kd 1.0 1.0 1.0\n")
            mtl_file.write("Ks 0.0 0.0 0.0\n")
            mtl_file.write("d 1.0\n")
            mtl_file.write("illum 1\n")
            if sprite_texture_path and os.path.exists(texture_path):
                mtl_file.write(f"map_Kd {texture_name}\n")
        
        # === GREEDY MESHER ===
        # Reuse the same greedy meshing logic from extrude_sprite_to_voxel_obj
        
        vertices_dict = {}
        vertices = []
        faces = []
        v_idx = 1
        hs = 0.5
        
        def get_vertex_id(vx, vy, vz):
            nonlocal v_idx
            key = (round(vx, 4), round(vy, 4), round(vz, 4))
            if key not in vertices_dict:
                vertices_dict[key] = v_idx
                vertices.append(key)
                v_idx += 1
            return vertices_dict[key]
        
        uvs_dict = {}
        uvs = []
        uv_idx = 1
        
        def get_uv_id(u, v):
            nonlocal uv_idx
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            key = (round(u, 6), round(v, 6))
            if key not in uvs_dict:
                uvs_dict[key] = uv_idx
                uvs.append(key)
                uv_idx += 1
            return uvs_dict[key]
        
        def greedy_mesh_2d(mask_2d):
            h_dim, w_dim = mask_2d.shape
            visited = np.zeros((h_dim, w_dim), dtype=bool)
            quads = []
            for i in range(h_dim):
                for j in range(w_dim):
                    if mask_2d[i, j] and not visited[i, j]:
                        w_q = 1
                        while j + w_q < w_dim and mask_2d[i, j + w_q] and not visited[i, j + w_q]:
                            w_q += 1
                        h_q = 1
                        can_expand = True
                        while i + h_q < h_dim and can_expand:
                            for w_idx in range(w_q):
                                if not mask_2d[i + h_q, j + w_idx] or visited[i + h_q, j + w_idx]:
                                    can_expand = False
                                    break
                            if can_expand:
                                h_q += 1
                        visited[i:i+h_q, j:j+w_q] = True
                        quads.append((j, i, w_q, h_q))
            return quads
        
        print("[*] Running Greedy Mesher on voxel grid...")
        
        # 1. Front Faces (+Z)
        for z in range(depth_size):
            if z == depth_size - 1:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z + 1]
            quads = greedy_mesh_2d(visible)
            z_val = z - depth_size / 2.0 + hs
            for (x, y, w, h) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                vt_bl = get_uv_id(x / grid_width, 1.0 - (y + h) / height)
                vt_br = get_uv_id((x + w) / grid_width, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                vt_tl = get_uv_id(x / grid_width, 1.0 - y / height)
                faces.append(f"{v_bl}/{vt_bl} {v_br}/{vt_br} {v_tr}/{vt_tr} {v_tl}/{vt_tl}")
        
        # 2. Back Faces (-Z)
        for z in range(depth_size):
            if z == 0:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z - 1]
            quads = greedy_mesh_2d(visible)
            z_val = z - depth_size / 2.0 - hs
            for (x, y, w, h) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                vt_bl = get_uv_id(x / grid_width, 1.0 - (y + h) / height)
                vt_br = get_uv_id((x + w) / grid_width, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                vt_tl = get_uv_id(x / grid_width, 1.0 - y / height)
                faces.append(f"{v_br}/{vt_br} {v_bl}/{vt_bl} {v_tl}/{vt_tl} {v_tr}/{vt_tr}")
        
        # 3. Right Faces (+X)
        for x in range(grid_width):
            if x == grid_width - 1:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x + 1, :]
            quads = greedy_mesh_2d(visible)
            x_val = x - grid_width / 2.0 + hs
            for (z, y, d, h) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                vt_bl = get_uv_id(z / depth_size, 1.0 - (y + h) / height)
                vt_br = get_uv_id((z + d) / depth_size, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((z + d) / depth_size, 1.0 - y / height)
                vt_tl = get_uv_id(z / depth_size, 1.0 - y / height)
                faces.append(f"{v_br}/{vt_br} {v_bl}/{vt_bl} {v_tl}/{vt_tl} {v_tr}/{vt_tr}")
        
        # 4. Left Faces (-X)
        for x in range(grid_width):
            if x == 0:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x - 1, :]
            quads = greedy_mesh_2d(visible)
            x_val = x - grid_width / 2.0 - hs
            for (z, y, d, h) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                vt_bl = get_uv_id(z / depth_size, 1.0 - (y + h) / height)
                vt_br = get_uv_id((z + d) / depth_size, 1.0 - (y + h) / height)
                vt_tr = get_uv_id((z + d) / depth_size, 1.0 - y / height)
                vt_tl = get_uv_id(z / depth_size, 1.0 - y / height)
                faces.append(f"{v_bl}/{vt_bl} {v_br}/{vt_br} {v_tr}/{vt_tr} {v_tl}/{vt_tl}")
        
        # 5. Top Faces (+Y)
        for y in range(height):
            if y == 0:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y - 1, :, :]
            quads = greedy_mesh_2d(visible)
            y_val = height / 2.0 - y + hs
            for (z, x, d, w) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                vt_min = get_uv_id(x / grid_width, 1.0 - y / height)
                vt_max = get_uv_id((x + w) / grid_width, 1.0 - y / height)
                faces.append(f"{v_bl}/{vt_min} {v_tl}/{vt_min} {v_tr}/{vt_max} {v_br}/{vt_max}")
        
        # 6. Bottom Faces (-Y)
        for y in range(height):
            if y == height - 1:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y + 1, :, :]
            quads = greedy_mesh_2d(visible)
            y_val = height / 2.0 - y - hs
            for (z, x, d, w) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                vt_min = get_uv_id(x / grid_width, 1.0 - (y + 1) / height)
                vt_max = get_uv_id((x + w) / grid_width, 1.0 - (y + 1) / height)
                faces.append(f"{v_bl}/{vt_min} {v_br}/{vt_max} {v_tr}/{vt_max} {v_tl}/{vt_min}")
        
        # Write OBJ file
        with open(output_obj_path, "w") as obj_file:
            obj_file.write(f"# Voxelized 3D Object: {base_name}\n")
            obj_file.write(f"mtllib {mtl_filename}\n\n")
            for vert in vertices:
                obj_file.write(f"v {vert[0]:.4f} {vert[1]:.4f} {vert[2]:.4f}\n")
            obj_file.write("\n")
            for uvc in uvs:
                obj_file.write(f"vt {uvc[0]:.6f} {uvc[1]:.6f}\n")
            obj_file.write("\n")
            obj_file.write(f"usemtl {base_name}_material\n")
            for face in faces:
                obj_file.write(f"f {face}\n")
        
        print(f"[+] Voxelized 3D Model saved to {output_obj_path} ({len(vertices)} vertices, {len(faces)} faces)")
        return True
    except Exception as e:
        print(f"[!] Error voxelizing mesh: {e}")
        import traceback
        traceback.print_exc()
        return False


def voxelize_from_primitives(primitives, output_obj_path, grid_size=32):
    """
    Build a voxel .obj from LLM-generated box primitives with per-face vertex colors.
    
    This completely bypasses image-to-3D reconstruction. The LLM describes the object
    as simple boxes, we fill a voxel grid, and greedy-mesh it into a clean .obj.
    
    Args:
        primitives: List of dicts with 'pos', 'size', 'color' keys
        output_obj_path: Path to write the .obj file
        grid_size: Size of the cubic voxel grid (default 32)
    """
    try:
        from voxel_builder import build_voxel_grid_from_primitives, hex_to_rgb
        
        print(f"[*] Building voxel grid from {len(primitives)} primitives...")
        grid, color_grid = build_voxel_grid_from_primitives(primitives, grid_size)
        
        if np.sum(grid) == 0:
            print("[!] No voxels were filled!")
            return False
        
        # The grid from voxel_builder is (Y, X, Z) - Y=0 is bottom
        # Our greedy mesher expects (height/Y, width/X, depth/Z) with Y=0 at TOP
        # So we need to flip Y axis
        grid = np.flip(grid, axis=0)
        color_grid = np.flip(color_grid, axis=0)
        
        height, grid_width, depth_size = grid.shape
        
        # Prepare file names
        base_dir = os.path.dirname(output_obj_path)
        base_name = os.path.splitext(os.path.basename(output_obj_path))[0]
        mtl_filename = f"{base_name}.mtl"
        output_mtl_path = os.path.join(base_dir, mtl_filename)
        
        # === COLLECT UNIQUE COLORS AND CREATE MATERIALS ===
        # Find all unique colors in the color grid where voxels exist
        color_materials = {}  # (r,g,b) -> material_name
        mat_idx = 0
        
        for y in range(height):
            for x in range(grid_width):
                for z in range(depth_size):
                    if grid[y, x, z]:
                        c = tuple(color_grid[y, x, z])
                        if c not in color_materials:
                            color_materials[c] = f"mat_{mat_idx}"
                            mat_idx += 1
        
        # Write MTL file with all color materials
        with open(output_mtl_path, "w") as mtl_file:
            mtl_file.write(f"# Materials for {base_name}\n\n")
            for (r, g, b), mat_name in color_materials.items():
                mtl_file.write(f"newmtl {mat_name}\n")
                mtl_file.write(f"Ka {r/255:.3f} {g/255:.3f} {b/255:.3f}\n")
                mtl_file.write(f"Kd {r/255:.3f} {g/255:.3f} {b/255:.3f}\n")
                mtl_file.write("Ks 0.1 0.1 0.1\n")
                mtl_file.write("d 1.0\n")
                mtl_file.write("illum 2\n\n")
        
        # === GREEDY MESHER WITH PER-FACE COLORS ===
        vertices_dict = {}
        vertices = []
        # faces_by_material: material_name -> list of face strings
        faces_by_material = {mat: [] for mat in color_materials.values()}
        v_idx = 1
        hs = 0.5
        
        def get_vertex_id(vx, vy, vz):
            nonlocal v_idx
            key = (round(vx, 4), round(vy, 4), round(vz, 4))
            if key not in vertices_dict:
                vertices_dict[key] = v_idx
                vertices.append(key)
                v_idx += 1
            return vertices_dict[key]
        
        def get_face_color(face_mask, axis_idx, slice_idx):
            """Get the dominant color for a visible face slice."""
            # Find the first True voxel in this face to get its color
            coords = np.argwhere(face_mask)
            if len(coords) == 0:
                return (128, 128, 128)
            # Use the color of the first visible voxel
            i, j = coords[0]
            if axis_idx == 0:  # Z-axis faces
                return tuple(color_grid[i, j, slice_idx])
            elif axis_idx == 1:  # X-axis faces
                return tuple(color_grid[i, slice_idx, j])
            else:  # Y-axis faces
                return tuple(color_grid[slice_idx, i, j])
        
        def greedy_mesh_2d_colored(mask_2d, color_slice_2d):
            """Greedy mesh that groups by both geometry AND color."""
            h_dim, w_dim = mask_2d.shape
            visited = np.zeros((h_dim, w_dim), dtype=bool)
            quads = []  # (j, i, w_q, h_q, color_tuple)
            for i in range(h_dim):
                for j in range(w_dim):
                    if mask_2d[i, j] and not visited[i, j]:
                        cur_color = tuple(color_slice_2d[i, j])
                        w_q = 1
                        while (j + w_q < w_dim and mask_2d[i, j + w_q] 
                               and not visited[i, j + w_q]
                               and tuple(color_slice_2d[i, j + w_q]) == cur_color):
                            w_q += 1
                        h_q = 1
                        can_expand = True
                        while i + h_q < h_dim and can_expand:
                            for w_idx in range(w_q):
                                if (not mask_2d[i + h_q, j + w_idx] 
                                    or visited[i + h_q, j + w_idx]
                                    or tuple(color_slice_2d[i + h_q, j + w_idx]) != cur_color):
                                    can_expand = False
                                    break
                            if can_expand:
                                h_q += 1
                        visited[i:i+h_q, j:j+w_q] = True
                        quads.append((j, i, w_q, h_q, cur_color))
            return quads
        
        print("[*] Running Color-Aware Greedy Mesher...")
        
        # 1. Front Faces (+Z)
        for z in range(depth_size):
            if z == depth_size - 1:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z + 1]
            color_slice = color_grid[:, :, z]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            z_val = z - depth_size / 2.0 + hs
            for (x, y, w, h, color) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_bl} {v_br} {v_tr} {v_tl}")
        
        # 2. Back Faces (-Z)
        for z in range(depth_size):
            if z == 0:
                visible = grid[:, :, z]
            else:
                visible = grid[:, :, z] & ~grid[:, :, z - 1]
            color_slice = color_grid[:, :, z]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            z_val = z - depth_size / 2.0 - hs
            for (x, y, w, h, color) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                v_bl = get_vertex_id(x_min, y_min, z_val)
                v_br = get_vertex_id(x_max, y_min, z_val)
                v_tr = get_vertex_id(x_max, y_max, z_val)
                v_tl = get_vertex_id(x_min, y_max, z_val)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_br} {v_bl} {v_tl} {v_tr}")
        
        # 3. Right Faces (+X)
        for x in range(grid_width):
            if x == grid_width - 1:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x + 1, :]
            color_slice = color_grid[:, x, :]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            x_val = x - grid_width / 2.0 + hs
            for (z, y, d, h, color) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_br} {v_bl} {v_tl} {v_tr}")
        
        # 4. Left Faces (-X)
        for x in range(grid_width):
            if x == 0:
                visible = grid[:, x, :]
            else:
                visible = grid[:, x, :] & ~grid[:, x - 1, :]
            color_slice = color_grid[:, x, :]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            x_val = x - grid_width / 2.0 - hs
            for (z, y, d, h, color) in quads:
                y_max = height / 2.0 - y + hs
                y_min = height / 2.0 - (y + h - 1) - hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_val, y_min, z_min)
                v_br = get_vertex_id(x_val, y_min, z_max)
                v_tr = get_vertex_id(x_val, y_max, z_max)
                v_tl = get_vertex_id(x_val, y_max, z_min)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_bl} {v_br} {v_tr} {v_tl}")
        
        # 5. Top Faces (+Y)
        for y in range(height):
            if y == 0:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y - 1, :, :]
            color_slice = color_grid[y, :, :]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            y_val = height / 2.0 - y + hs
            for (z, x, d, w, color) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_bl} {v_tl} {v_tr} {v_br}")
        
        # 6. Bottom Faces (-Y)
        for y in range(height):
            if y == height - 1:
                visible = grid[y, :, :]
            else:
                visible = grid[y, :, :] & ~grid[y + 1, :, :]
            color_slice = color_grid[y, :, :]
            quads = greedy_mesh_2d_colored(visible, color_slice)
            y_val = height / 2.0 - y - hs
            for (z, x, d, w, color) in quads:
                x_min = x - grid_width / 2.0 - hs
                x_max = (x + w - 1) - grid_width / 2.0 + hs
                z_min = z - depth_size / 2.0 - hs
                z_max = (z + d - 1) - depth_size / 2.0 + hs
                v_bl = get_vertex_id(x_min, y_val, z_min)
                v_br = get_vertex_id(x_max, y_val, z_min)
                v_tr = get_vertex_id(x_max, y_val, z_max)
                v_tl = get_vertex_id(x_min, y_val, z_max)
                mat = color_materials[color]
                faces_by_material[mat].append(f"{v_bl} {v_br} {v_tr} {v_tl}")
        
        # Write OBJ file
        total_faces = sum(len(f) for f in faces_by_material.values())
        with open(output_obj_path, "w") as obj_file:
            obj_file.write(f"# LLM-Built Voxel Object: {base_name}\n")
            obj_file.write(f"mtllib {mtl_filename}\n\n")
            for vert in vertices:
                obj_file.write(f"v {vert[0]:.4f} {vert[1]:.4f} {vert[2]:.4f}\n")
            obj_file.write("\n")
            for mat_name, mat_faces in faces_by_material.items():
                if mat_faces:
                    obj_file.write(f"usemtl {mat_name}\n")
                    for face in mat_faces:
                        obj_file.write(f"f {face}\n")
                    obj_file.write("\n")
        
        print(f"[+] LLM-Built Voxel Model saved to {output_obj_path} ({len(vertices)} vertices, {total_faces} faces, {len(color_materials)} colors)")
        return True
    except Exception as e:
        print(f"[!] Error building voxel model from primitives: {e}")
        import traceback
        traceback.print_exc()
        return False
