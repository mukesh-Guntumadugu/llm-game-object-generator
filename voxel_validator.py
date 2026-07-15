"""
voxel_validator.py — Automated 3D Model Quality Judge

Renders the generated .obj from multiple camera angles using Blender headless,
then sends each render to a Vision LLM (llama3.2-vision) to verify the object
is recognizable. If the VLM can't identify it, the model fails validation.
"""

import os
import json
import subprocess
import tempfile
import base64
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Blender script to render OBJ from multiple angles
BLENDER_RENDER_SCRIPT = '''
import bpy
import sys
import math
import os

# Get arguments after "--"
argv = sys.argv[sys.argv.index("--") + 1:]
obj_path = argv[0]
output_dir = argv[1]
num_angles = int(argv[2]) if len(argv) > 2 else 4

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
bpy.ops.wm.obj_import(filepath=obj_path)

# Get the imported object(s)
imported = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not imported:
    print("ERROR: No mesh objects found in OBJ file")
    sys.exit(1)

# Select all imported and join
for obj in imported:
    obj.select_set(True)
bpy.context.view_layer.objects.active = imported[0]
if len(imported) > 1:
    bpy.ops.object.join()

target = bpy.context.active_object

# Center object at origin
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
target.location = (0, 0, 0)

# Calculate camera distance based on object size
dims = target.dimensions
max_dim = max(dims.x, dims.y, dims.z)
cam_distance = max_dim * 2.5

# Add camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object
bpy.context.scene.camera = camera

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0

bpy.ops.object.light_add(type='POINT', location=(-5, -5, 5))
fill = bpy.context.active_object
fill.data.energy = 500.0

# Set render settings
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Set white world background
world = bpy.data.worlds.new("RenderWorld")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 1.0

# Render from multiple angles
angles = []
for i in range(num_angles):
    angle = (2 * math.pi * i) / num_angles
    angles.append(angle)

for i, angle in enumerate(angles):
    # Position camera
    cx = cam_distance * math.sin(angle)
    cy = -cam_distance * math.cos(angle)
    cz = cam_distance * 0.5  # Slightly above
    camera.location = (cx, cy, cz)
    
    # Point camera at object center
    direction = target.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Render
    angle_name = ["front", "right", "back", "left"][i % 4] if num_angles == 4 else f"angle_{i}"
    output_path = os.path.join(output_dir, f"render_{angle_name}.png")
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")

print("RENDER_COMPLETE")
'''


def render_obj_multi_angle(obj_path, output_dir, num_angles=4):
    """
    Use Blender headless to render an OBJ file from multiple camera angles.
    Returns list of rendered image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write Blender script to temp file
    script_path = os.path.join(output_dir, "_render_script.py")
    with open(script_path, 'w') as f:
        f.write(BLENDER_RENDER_SCRIPT)
    
    print(f"[*] Rendering 3D model from {num_angles} angles...")
    
    try:
        result = subprocess.run(
            ["blender", "--background", "--python", script_path, "--", obj_path, output_dir, str(num_angles)],
            capture_output=True, text=True, timeout=60
        )
        
        if "RENDER_COMPLETE" not in result.stdout:
            print(f"[!] Blender render may have failed")
            if result.stderr:
                # Filter out noise
                errors = [l for l in result.stderr.split('\n') if 'ERROR' in l.upper()]
                for e in errors:
                    print(f"    {e}")
    except subprocess.TimeoutExpired:
        print("[!] Blender render timed out")
    except FileNotFoundError:
        print("[!] Blender not found in PATH")
    
    # Clean up script
    if os.path.exists(script_path):
        os.remove(script_path)
    
    # Collect rendered images
    renders = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir) 
        if f.startswith("render_") and f.endswith(".png")
    ])
    
    print(f"[+] Rendered {len(renders)} views")
    return renders


def ask_vlm_what_is_this(image_path, model="llama3.2-vision"):
    """
    Send a rendered image to the Vision LLM and ask it to identify the object.
    Returns the VLM's answer.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": "What 3D object is shown in this image? Answer with just the object name in 1-3 words. If you cannot identify it or it looks like random shapes, say 'unrecognizable'.",
                "images": [image_data],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip().lower()
    except Exception as e:
        print(f"[!] VLM query failed: {e}")
        return "error"


def validate_3d_model(obj_path, expected_object_name, output_dir=None, vlm_model="llama3.2-vision"):
    """
    Validate a 3D .obj model by rendering it from multiple angles and asking
    a Vision LLM to identify it.
    
    Args:
        obj_path: Path to the .obj file
        expected_object_name: What the object should be (e.g. "wooden chair")
        output_dir: Directory for render outputs (defaults to same as obj)
        vlm_model: Vision LLM model to use
        
    Returns:
        dict with:
            - passed: bool
            - score: float (0.0 - 1.0, fraction of angles correctly identified)
            - details: list of per-angle results
    """
    if output_dir is None:
        output_dir = os.path.dirname(obj_path)
    
    render_dir = os.path.join(output_dir, "_validation_renders")
    
    print(f"\n{'='*50}")
    print(f"[*] 3D MODEL VALIDATION: '{expected_object_name}'")
    print(f"{'='*50}")
    
    # Step 1: Render from 4 angles
    renders = render_obj_multi_angle(obj_path, render_dir, num_angles=4)
    
    if not renders:
        print("[!] No renders produced — validation cannot proceed")
        return {"passed": False, "score": 0.0, "details": []}
    
    # Step 2: Ask VLM to identify each angle
    expected_lower = expected_object_name.lower().strip()
    # Build list of acceptable keywords from the object name
    # Remove common prefixes and split compound words
    stripped = expected_lower
    for prefix in ["wooden ", "iron ", "steel ", "stone ", "gold ", "golden "]:
        stripped = stripped.replace(prefix, "")
    keywords = stripped.split()
    # Also add sub-words from compound words (e.g. "bookshelf" → "book", "shelf")
    extra = []
    for kw in keywords:
        if len(kw) > 6:
            # Common compound splits
            for split_at in range(3, len(kw) - 2):
                left, right = kw[:split_at], kw[split_at:]
                if len(left) >= 3 and len(right) >= 3:
                    extra.extend([left, right])
    keywords.extend(extra)
    keywords = list(set(keywords))  # deduplicate
    
    details = []
    correct = 0
    
    for render_path in renders:
        angle_name = os.path.basename(render_path).replace("render_", "").replace(".png", "")
        vlm_answer = ask_vlm_what_is_this(render_path, model=vlm_model)
        
        # Check if any keyword matches (bidirectional)
        is_correct = any(kw in vlm_answer for kw in keywords) or any(vlm_answer.rstrip('.') in kw for kw in keywords)
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"  [{angle_name:>6}] VLM says: '{vlm_answer}' {status}")
        details.append({
            "angle": angle_name,
            "vlm_answer": vlm_answer,
            "correct": is_correct
        })
    
    score = correct / len(renders) if renders else 0.0
    passed = score == 1.0  # STRICT: Pass ONLY if ALL angles are recognized
    
    print(f"\n[{'+'  if passed else '!'}] Validation {'PASSED' if passed else 'FAILED'}: {correct}/{len(renders)} angles recognized (score: {score:.0%})")
    
    # Clean up render images
    for r in renders:
        if os.path.exists(r):
            os.remove(r)
    if os.path.exists(render_dir):
        try:
            os.rmdir(render_dir)
        except OSError:
            pass
    
    return {"passed": passed, "score": score, "details": details}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a 3D voxel model")
    parser.add_argument("obj_path", help="Path to .obj file")
    parser.add_argument("expected_name", help="Expected object name")
    parser.add_argument("--vlm", default="llama3.2-vision")
    args = parser.parse_args()
    
    result = validate_3d_model(args.obj_path, args.expected_name, vlm_model=args.vlm)
    print(f"\nResult: {json.dumps(result, indent=2)}")
