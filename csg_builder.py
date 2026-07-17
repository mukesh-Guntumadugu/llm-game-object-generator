import os
import json
import trimesh
import numpy as np
import requests
import re

def query_llm_for_csg(prompt, model="qwen2.5-coder:7b-instruct-fp16"):
    """
    Asks the LLM to design the 3D object using a list of geometric primitives.
    """
    sys_prompt = """You are an expert 3D modeler using Constructive Solid Geometry (CSG).
Your task is to break down the user's requested object into a list of basic geometric primitives.
Available primitives: "box", "sphere", "cylinder", "cone".

You must output ONLY valid JSON. No markdown, no explanation.

JSON Schema:
[
  {
    "type": "primitive_type",
    "dimensions": [length, width, height] OR [radius, height] for cylinders/cones OR [radius] for spheres,
    "translation": [x, y, z],
    "rotation": [rx, ry, rz] (in degrees, Euler angles),
    "color": [r, g, b] (float 0.0 to 1.0),
    "name": "part_name"
  }
]

Example for a Simple Sword:
[
  {"type": "cylinder", "dimensions": [0.1, 1.5], "translation": [0, 0, 0], "rotation": [0, 0, 0], "color": [0.8, 0.8, 0.8], "name": "blade"},
  {"type": "box", "dimensions": [0.8, 0.1, 0.2], "translation": [0, -0.75, 0], "rotation": [0, 0, 0], "color": [0.8, 0.6, 0.1], "name": "crossguard"},
  {"type": "cylinder", "dimensions": [0.08, 0.4], "translation": [0, -1.0, 0], "rotation": [0, 0, 0], "color": [0.3, 0.2, 0.1], "name": "grip"},
  {"type": "sphere", "dimensions": [0.15], "translation": [0, -1.25, 0], "rotation": [0, 0, 0], "color": [0.8, 0.6, 0.1], "name": "pommel"}
]

Make the object look good. Ensure translations connect the parts logically.
"""

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": sys_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        output_text = response.json().get("response", "")
        
        # Extract JSON
        json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except Exception as e:
        print(f"[!] LLM Request failed: {e}")
        return None

def euler_to_matrix(rx, ry, rz):
    """Converts degrees to a 4x4 rotation matrix."""
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)
    
    Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    
    return trimesh.transformations.concatenate_matrices(Rz, Ry, Rx)

def build_csg_mesh(prompt, output_path, llm_model="qwen2.5-coder:7b-instruct-fp16"):
    """
    Builds the final .obj by parsing LLM JSON and combining high-poly trimesh primitives.
    """
    print(f"[*] Asking LLM to design CSG layout for: '{prompt}'...")
    primitives_data = query_llm_for_csg(prompt, model=llm_model)
    
    if not primitives_data:
        print("[!] Failed to get valid CSG JSON from LLM.")
        return None
        
    print(f"[*] Parsed {len(primitives_data)} geometric primitives from LLM. Assembling...")
    
    meshes = []
    
    for p in primitives_data:
        ptype = p.get("type", "").lower()
        dims = p.get("dimensions", [])
        trans = p.get("translation", [0, 0, 0])
        rot = p.get("rotation", [0, 0, 0])
        color = p.get("color", [0.5, 0.5, 0.5])
        
        # Ensure alpha channel for color
        if len(color) == 3:
            color = [int(c * 255) for c in color] + [255]
            
        mesh = None
        # High polygon count for smooth geometry
        try:
            if ptype == "box" and len(dims) == 3:
                mesh = trimesh.creation.box(extents=dims)
            elif ptype == "sphere" and len(dims) >= 1:
                mesh = trimesh.creation.icosphere(subdivisions=4, radius=dims[0])
            elif ptype == "cylinder" and len(dims) >= 2:
                mesh = trimesh.creation.cylinder(radius=dims[0], height=dims[1], sections=64)
            elif ptype == "cone" and len(dims) >= 2:
                mesh = trimesh.creation.cone(radius=dims[0], height=dims[1], sections=64)
            else:
                print(f"[!] Warning: Unknown primitive type or bad dimensions: {p}")
                continue
        except Exception as e:
            print(f"[!] Error creating primitive {ptype}: {e}")
            continue
            
        if mesh:
            # Apply color
            mesh.visual.vertex_colors = color
            
            # Apply Rotation
            rot_matrix = euler_to_matrix(rot[0], rot[1], rot[2])
            mesh.apply_transform(rot_matrix)
            
            # Apply Translation
            trans_matrix = trimesh.transformations.translation_matrix(trans)
            mesh.apply_transform(trans_matrix)
            
            meshes.append(mesh)

    if not meshes:
        print("[!] No valid meshes were created.")
        return None
        
    # Combine into a single mesh
    print("[*] Fusing primitives into single high-resolution mesh...")
    combined = trimesh.util.concatenate(meshes)
    
    # Export to OBJ
    print(f"[*] Exporting to {output_path}...")
    combined.export(output_path)
    
    print(f"[+] LLM CSG generation complete! ({len(combined.vertices)} vertices, {len(combined.faces)} faces)")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        build_csg_mesh(sys.argv[1], sys.argv[2])
    else:
        build_csg_mesh("A medieval broadsword with a golden hilt and long steel blade", "test_csg_sword.obj")
