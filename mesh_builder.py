import os
import math

def build_apple_mesh(output_path):
    """
    Builds a high-fidelity parametric 3D Apple mesh with >20k triangles,
    a distinct curved leaf, and per-face procedural shading.
    """
    vertices = []
    normals = []
    faces = []
    materials = {}

    # Mesh Density
    segments_theta = 96
    segments_phi = 128

    # --- 1. Apple Body Geometry ---
    body_start_idx = len(vertices) + 1
    for i in range(segments_theta + 1):
        theta = i * math.pi / segments_theta
        # Parametric shape: Sphere with dimples at top (theta=0) and bottom (theta=pi)
        dimple = 1.0 - 0.3 * math.exp(-20 * theta**2) - 0.2 * math.exp(-15 * (math.pi - theta)**2)
        r = dimple * 0.9  # Scale radius

        for j in range(segments_phi):
            phi = j * 2 * math.pi / segments_phi
            # Subtle asymmetry
            radius_mod = r * (1.0 + 0.05 * math.sin(4 * phi) * math.sin(theta))
            x = radius_mod * math.sin(theta) * math.cos(phi)
            y = radius_mod * math.cos(theta)
            z = radius_mod * math.sin(theta) * math.sin(phi)

            vertices.append((x, y, z))
            
            # Simple spherical normal approximation for the body
            mag = math.sqrt(x*x + y*y + z*z)
            if mag == 0: mag = 1
            normals.append((x/mag, y/mag, z/mag))

    # Construct Faces for Body
    for i in range(segments_theta):
        for j in range(segments_phi):
            p1 = body_start_idx + i * segments_phi + j
            p2 = body_start_idx + i * segments_phi + (j + 1) % segments_phi
            p3 = body_start_idx + (i + 1) * segments_phi + j
            p4 = body_start_idx + (i + 1) * segments_phi + (j + 1) % segments_phi

            # Material per face based on height (y) and angle (phi)
            y_val = vertices[p1 - 1][1]
            phi_val = j * 2 * math.pi / segments_phi
            
            # Procedural shading: base red with yellow highlights and shadow mapping
            red = 0.8 + 0.2 * math.sin(phi_val)
            green = 0.1 + 0.3 * (y_val + 1.0) / 2.0
            blue = 0.1 + 0.1 * math.cos(phi_val * 2)
            
            # Clamp
            red = max(0.0, min(1.0, red))
            green = max(0.0, min(1.0, green))
            blue = max(0.0, min(1.0, blue))
            
            mat_name = f"MatBody_{i}_{j}"
            materials[mat_name] = (red, green, blue)

            faces.append((p1, p3, p2, mat_name))
            faces.append((p2, p3, p4, mat_name))

    # --- 2. Stem Geometry ---
    stem_start_idx = len(vertices) + 1
    stem_segments = 12
    stem_rings = 8
    
    for i in range(stem_rings + 1):
        h = i / stem_rings
        y = 0.7 + h * 0.4
        r = 0.03 * (1.0 - 0.2 * h)
        # Curve stem
        x_offset = 0.1 * h**2
        z_offset = 0.05 * h**2

        for j in range(stem_segments):
            angle = j * 2 * math.pi / stem_segments
            x = x_offset + r * math.cos(angle)
            z = z_offset + r * math.sin(angle)
            vertices.append((x, y, z))
            normals.append((math.cos(angle), 0, math.sin(angle)))

    for i in range(stem_rings):
        for j in range(stem_segments):
            p1 = stem_start_idx + i * stem_segments + j
            p2 = stem_start_idx + i * stem_segments + (j + 1) % stem_segments
            p3 = stem_start_idx + (i + 1) * stem_segments + j
            p4 = stem_start_idx + (i + 1) * stem_segments + (j + 1) % stem_segments
            
            mat_name = f"MatStem_{i}"
            if mat_name not in materials:
                materials[mat_name] = (0.3, 0.2, 0.1) # Brown stem
            
            faces.append((p1, p3, p2, mat_name))
            faces.append((p2, p3, p4, mat_name))

    # --- 3. Leaf Geometry ---
    leaf_start_idx = len(vertices) + 1
    leaf_segments_u = 10
    leaf_segments_v = 6
    
    # Base of leaf attaches to top of stem
    leaf_base = vertices[stem_start_idx + stem_rings * stem_segments - 1]
    
    for u in range(leaf_segments_u + 1):
        u_norm = u / leaf_segments_u
        for v in range(leaf_segments_v + 1):
            v_norm = v / leaf_segments_v - 0.5
            
            # Parametric leaf shape
            width = 0.4 * math.sin(u_norm * math.pi)
            length = 0.6 * u_norm
            
            # Curve and droop
            droop = -0.3 * (u_norm**2)
            
            x = leaf_base[0] + length * 0.8 + width * v_norm * 0.2
            y = leaf_base[1] + droop + width * v_norm * 0.1
            z = leaf_base[2] + width * v_norm * 0.9
            
            vertices.append((x, y, z))
            normals.append((0, 1, 0)) # simplified normal

    for u in range(leaf_segments_u):
        for v in range(leaf_segments_v):
            p1 = leaf_start_idx + u * (leaf_segments_v + 1) + v
            p2 = leaf_start_idx + u * (leaf_segments_v + 1) + (v + 1)
            p3 = leaf_start_idx + (u + 1) * (leaf_segments_v + 1) + v
            p4 = leaf_start_idx + (u + 1) * (leaf_segments_v + 1) + (v + 1)
            
            mat_name = f"MatLeaf_{u}_{v}"
            if mat_name not in materials:
                materials[mat_name] = (0.1, 0.6 + 0.2 * u_norm, 0.1) # Green leaf
            
            faces.append((p1, p3, p2, mat_name))
            faces.append((p2, p3, p4, mat_name))

    # --- Write OBJ and MTL ---
    mtl_filename = output_path.replace('.obj', '.mtl')
    
    # Write MTL
    with open(mtl_filename, 'w') as f:
        for mat_name, color in materials.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            f.write(f"Ks 0.1 0.1 0.1\n")
            f.write(f"Ns 10\n\n")

    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        for vn in normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            
        current_mat = None
        for face in faces:
            v1, v2, v3, mat = face
            if mat != current_mat:
                f.write(f"usemtl {mat}\n")
                current_mat = mat
            
            # OBJ is 1-indexed, using vertex normals
            f.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
            
    print(f"[+] Parametric Apple Mesh generated successfully: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")
    return output_path

def build_parametric_mesh(prompt, output_path):
    """
    Entry point for the parametric mesh generator.
    Parses the prompt and calls the appropriate mathematical builder.
    """
    prompt_lower = prompt.lower()
    if 'apple' in prompt_lower:
        return build_apple_mesh(output_path)
    # Extensible to other shapes (sword, chest, etc.)
    return None
