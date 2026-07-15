"""
voxel_builder.py — LLM-Driven Voxel Construction

Instead of reconstructing 3D from 2D images (which always fails),
we ask the LLM to describe objects as 3D box primitives with positions,
sizes, and colors. This produces structurally perfect voxel objects.
"""

import os
import json
import re
import numpy as np
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# System prompt that tells the LLM how to describe 3D objects
VOXEL_SYSTEM_PROMPT = """You are a 3D voxel artist. Given an object name, you must describe it as a collection of 3D rectangular box primitives that together form the shape of that object.

RULES:
1. The object must fit within a 48x48x48 grid (coordinates 0-47)
2. Y=0 is the BOTTOM (ground level), Y=47 is the top
3. X is left-right, Z is front-back
4. Each primitive has: name, position [x,y,z] (bottom-left-front corner), size [width,height,depth], and color (hex)
5. Use realistic proportions - a chair seat should be flat and wide, legs should be tall and thin
6. Center the object in the grid horizontally (X and Z)
7. Place the object on the ground (Y=0 for the lowest point)
8. Use 15-30 primitives for rich detail: include structural parts, trim, decorative elements, edges
9. Colors should be realistic with subtle variations (e.g. use 3-5 shades of the same material)
10. Add small details: beveled edges, trim pieces, fasteners, caps on legs, decorative strips

You MUST respond with ONLY valid JSON, no explanation text. Use this exact format:
{"primitives": [{"name": "part_name", "pos": [x, y, z], "size": [w, h, d], "color": "#RRGGBB"}, ...]}"""


EXAMPLE_OBJECTS = {
    "wooden chair": {
        "primitives": [
            # Seat
            {"name": "seat_top", "pos": [8, 20, 8], "size": [32, 2, 32], "color": "#8B4513"},
            {"name": "seat_underside", "pos": [9, 18, 9], "size": [30, 2, 30], "color": "#7A3B10"},
            {"name": "seat_front_lip", "pos": [8, 19, 7], "size": [32, 3, 1], "color": "#6B3410"},
            {"name": "seat_left_lip", "pos": [7, 19, 8], "size": [1, 3, 32], "color": "#6B3410"},
            {"name": "seat_right_lip", "pos": [40, 19, 8], "size": [1, 3, 32], "color": "#6B3410"},
            # Front legs
            {"name": "leg_front_left", "pos": [9, 0, 9], "size": [4, 18, 4], "color": "#A0522D"},
            {"name": "leg_front_right", "pos": [35, 0, 9], "size": [4, 18, 4], "color": "#A0522D"},
            {"name": "leg_fl_cap", "pos": [9, 0, 9], "size": [4, 1, 4], "color": "#5C3317"},
            {"name": "leg_fr_cap", "pos": [35, 0, 9], "size": [4, 1, 4], "color": "#5C3317"},
            # Back legs (taller, extend up into backrest)
            {"name": "leg_back_left", "pos": [9, 0, 35], "size": [4, 42, 4], "color": "#A0522D"},
            {"name": "leg_back_right", "pos": [35, 0, 35], "size": [4, 42, 4], "color": "#A0522D"},
            {"name": "leg_bl_cap", "pos": [9, 0, 35], "size": [4, 1, 4], "color": "#5C3317"},
            {"name": "leg_br_cap", "pos": [35, 0, 35], "size": [4, 1, 4], "color": "#5C3317"},
            # Cross braces between legs
            {"name": "brace_front", "pos": [13, 6, 10], "size": [22, 2, 2], "color": "#6B3410"},
            {"name": "brace_left", "pos": [10, 6, 13], "size": [2, 2, 22], "color": "#6B3410"},
            {"name": "brace_right", "pos": [36, 6, 13], "size": [2, 2, 22], "color": "#6B3410"},
            {"name": "brace_back", "pos": [13, 6, 36], "size": [22, 2, 2], "color": "#6B3410"},
            # Backrest
            {"name": "backrest_top_rail", "pos": [9, 40, 36], "size": [30, 2, 3], "color": "#8B4513"},
            {"name": "backrest_mid_rail", "pos": [9, 30, 36], "size": [30, 2, 3], "color": "#7A3B10"},
            {"name": "backrest_slat_1", "pos": [14, 22, 37], "size": [3, 18, 2], "color": "#A0522D"},
            {"name": "backrest_slat_2", "pos": [22, 22, 37], "size": [3, 18, 2], "color": "#A0522D"},
            {"name": "backrest_slat_3", "pos": [30, 22, 37], "size": [3, 18, 2], "color": "#A0522D"},
            # Decorative finials on top of back legs
            {"name": "finial_left", "pos": [9, 42, 35], "size": [4, 2, 4], "color": "#5C3317"},
            {"name": "finial_right", "pos": [35, 42, 35], "size": [4, 2, 4], "color": "#5C3317"}
        ]
    },
    "wooden table": {
        "primitives": [
            # Tabletop
            {"name": "tabletop", "pos": [4, 28, 4], "size": [40, 3, 40], "color": "#8B4513"},
            {"name": "tabletop_edge_front", "pos": [3, 27, 3], "size": [42, 1, 1], "color": "#6B3410"},
            {"name": "tabletop_edge_back", "pos": [3, 27, 44], "size": [42, 1, 1], "color": "#6B3410"},
            {"name": "tabletop_edge_left", "pos": [3, 27, 4], "size": [1, 1, 40], "color": "#6B3410"},
            {"name": "tabletop_edge_right", "pos": [44, 27, 4], "size": [1, 1, 40], "color": "#6B3410"},
            # Legs
            {"name": "leg_fl", "pos": [6, 0, 6], "size": [4, 27, 4], "color": "#A0522D"},
            {"name": "leg_fr", "pos": [38, 0, 6], "size": [4, 27, 4], "color": "#A0522D"},
            {"name": "leg_bl", "pos": [6, 0, 38], "size": [4, 27, 4], "color": "#A0522D"},
            {"name": "leg_br", "pos": [38, 0, 38], "size": [4, 27, 4], "color": "#A0522D"},
            # Leg caps
            {"name": "cap_fl", "pos": [6, 0, 6], "size": [4, 1, 4], "color": "#5C3317"},
            {"name": "cap_fr", "pos": [38, 0, 6], "size": [4, 1, 4], "color": "#5C3317"},
            {"name": "cap_bl", "pos": [6, 0, 38], "size": [4, 1, 4], "color": "#5C3317"},
            {"name": "cap_br", "pos": [38, 0, 38], "size": [4, 1, 4], "color": "#5C3317"},
            # Apron (under tabletop)
            {"name": "apron_front", "pos": [10, 22, 6], "size": [28, 5, 2], "color": "#7A3B10"},
            {"name": "apron_back", "pos": [10, 22, 40], "size": [28, 5, 2], "color": "#7A3B10"},
            {"name": "apron_left", "pos": [6, 22, 10], "size": [2, 5, 28], "color": "#7A3B10"},
            {"name": "apron_right", "pos": [40, 22, 10], "size": [2, 5, 28], "color": "#7A3B10"},
            # Cross braces
            {"name": "brace_front", "pos": [10, 10, 7], "size": [28, 2, 2], "color": "#6B3410"},
            {"name": "brace_back", "pos": [10, 10, 39], "size": [28, 2, 2], "color": "#6B3410"},
            {"name": "brace_left", "pos": [7, 10, 10], "size": [2, 2, 28], "color": "#6B3410"},
            {"name": "brace_right", "pos": [39, 10, 10], "size": [2, 2, 28], "color": "#6B3410"}
        ]
    },
    "sword": {
        "primitives": [
            # Blade
            {"name": "blade_core", "pos": [21, 16, 22], "size": [6, 28, 3], "color": "#C0C0C0"},
            {"name": "blade_edge_l", "pos": [20, 18, 22], "size": [1, 24, 3], "color": "#D8D8D8"},
            {"name": "blade_edge_r", "pos": [27, 18, 22], "size": [1, 24, 3], "color": "#D8D8D8"},
            {"name": "blade_tip_1", "pos": [22, 44, 22], "size": [4, 2, 3], "color": "#D4D4D4"},
            {"name": "blade_tip_2", "pos": [23, 46, 23], "size": [2, 1, 1], "color": "#E0E0E0"},
            {"name": "blade_fuller", "pos": [23, 18, 23], "size": [2, 22, 1], "color": "#A8A8A8"},
            # Guard
            {"name": "guard_main", "pos": [14, 13, 21], "size": [20, 3, 5], "color": "#DAA520"},
            {"name": "guard_curl_l", "pos": [13, 12, 22], "size": [2, 2, 3], "color": "#B8860B"},
            {"name": "guard_curl_r", "pos": [33, 12, 22], "size": [2, 2, 3], "color": "#B8860B"},
            {"name": "guard_gem", "pos": [23, 14, 23], "size": [2, 2, 1], "color": "#FF0000"},
            # Handle
            {"name": "handle_core", "pos": [21, 3, 22], "size": [6, 10, 4], "color": "#654321"},
            {"name": "handle_wrap_1", "pos": [21, 4, 22], "size": [6, 1, 4], "color": "#3B2010"},
            {"name": "handle_wrap_2", "pos": [21, 6, 22], "size": [6, 1, 4], "color": "#3B2010"},
            {"name": "handle_wrap_3", "pos": [21, 8, 22], "size": [6, 1, 4], "color": "#3B2010"},
            {"name": "handle_wrap_4", "pos": [21, 10, 22], "size": [6, 1, 4], "color": "#3B2010"},
            # Pommel
            {"name": "pommel", "pos": [20, 0, 21], "size": [8, 3, 6], "color": "#DAA520"},
            {"name": "pommel_gem", "pos": [23, 1, 23], "size": [2, 1, 2], "color": "#4169E1"}
        ]
    }
}


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple (0-255)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def build_voxel_grid_from_primitives(primitives, grid_size=32):
    """
    Build a 3D boolean grid and a 3D color grid from box primitives.
    
    Args:
        primitives: List of dicts with 'pos', 'size', 'color' keys
        grid_size: Size of the cubic grid
        
    Returns:
        grid: boolean numpy array (grid_size, grid_size, grid_size)
        color_grid: uint8 numpy array (grid_size, grid_size, grid_size, 3) for RGB
    """
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    color_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.uint8)
    
    for prim in primitives:
        pos = prim['pos']
        size = prim['size']
        color = hex_to_rgb(prim.get('color', '#888888'))
        
        x0, y0, z0 = int(pos[0]), int(pos[1]), int(pos[2])
        w, h, d = int(size[0]), int(size[1]), int(size[2])
        
        # Clamp to grid bounds
        x1 = min(x0 + w, grid_size)
        y1 = min(y0 + h, grid_size)
        z1 = min(z0 + d, grid_size)
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        z0 = max(z0, 0)
        
        grid[y0:y1, x0:x1, z0:z1] = True
        color_grid[y0:y1, x0:x1, z0:z1] = color
        
        print(f"  [+] {prim.get('name', 'unnamed')}: pos=({x0},{y0},{z0}) size=({w},{h},{d}) color={prim.get('color', '#888888')}")
    
    total = np.sum(grid)
    print(f"[*] Voxel grid built: {total} filled voxels in {grid_size}x{grid_size}x{grid_size} grid")
    return grid, color_grid


def query_llm_for_primitives(object_name, model="qwen2.5-coder:7b-instruct-fp16"):
    """
    Ask the LLM to describe an object as 3D box primitives.
    Falls back to built-in examples if LLM is unavailable.
    """
    # Check if we have a built-in example first (instant, no LLM needed)
    object_lower = object_name.lower().strip()
    for key, value in EXAMPLE_OBJECTS.items():
        if key in object_lower or object_lower in key:
            print(f"[*] Using built-in voxel template for '{key}'")
            return value['primitives']
    
    # Try querying the LLM
    print(f"[*] Querying LLM ({model}) for 3D primitives of '{object_name}'...")
    
    prompt = f"""{VOXEL_SYSTEM_PROMPT}

Here is an example for "wooden chair":
{json.dumps(EXAMPLE_OBJECTS["wooden chair"], indent=2)}

Now generate the primitives JSON for: "{object_name}"
Respond with ONLY the JSON object, nothing else."""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048}
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json().get("response", "")
        
        # Extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            primitives = parsed.get("primitives", [])
            if primitives and len(primitives) > 0:
                print(f"[+] LLM generated {len(primitives)} primitives for '{object_name}'")
                # Validate primitives
                valid = []
                for p in primitives:
                    if 'pos' in p and 'size' in p and len(p['pos']) == 3 and len(p['size']) == 3:
                        if 'color' not in p:
                            p['color'] = '#888888'
                        if 'name' not in p:
                            p['name'] = 'block'
                        valid.append(p)
                if valid:
                    return valid
        
        print(f"[!] LLM response could not be parsed as valid primitives JSON")
    except Exception as e:
        print(f"[!] LLM query failed: {e}")
    
    # Final fallback: generate a simple cube
    print(f"[!] Falling back to simple cube for '{object_name}'")
    return [
        {"name": "body", "pos": [8, 0, 8], "size": [16, 16, 16], "color": "#888888"}
    ]


def refine_primitives_with_feedback(object_name, previous_primitives, validation_details, model="qwen2.5-coder:7b-instruct-fp16"):
    """
    Feed VLM validation feedback back to the LLM to fix the 3D model.
    
    The VLM told us what each angle looks like (e.g. "from the front it looks 
    like a bed, not a chair"). We send this criticism to the LLM along with 
    the previous primitives so it can fix the structural issues.
    
    Args:
        object_name: Target object name
        previous_primitives: The primitives that failed validation
        validation_details: List of dicts with 'angle', 'vlm_answer', 'correct'
        model: LLM model to use
        
    Returns:
        List of improved primitive dicts
    """
    # Build feedback description
    feedback_lines = []
    for detail in validation_details:
        angle = detail["angle"]
        vlm_answer = detail["vlm_answer"]
        correct = detail["correct"]
        if not correct:
            feedback_lines.append(f"- From the {angle}, it looks like a '{vlm_answer}' instead of a '{object_name}'")
        else:
            feedback_lines.append(f"- From the {angle}, it correctly looks like a '{object_name}' ✓")
    
    feedback_text = "\n".join(feedback_lines)
    
    print(f"[*] Sending VLM feedback to LLM for correction...")
    print(f"    Feedback:\n{feedback_text}")
    
    prompt = f"""{VOXEL_SYSTEM_PROMPT}

I previously generated these primitives for "{object_name}":
{json.dumps(previous_primitives, indent=2)}

But when a Vision AI judged the 3D model from multiple camera angles, it found problems:
{feedback_text}

Please FIX the primitives so the object is clearly recognizable as a "{object_name}" from ALL angles. 
Think about what structural features make a {object_name} recognizable:
- What parts are missing or wrong?
- What proportions need to change?
- What details would make it more distinctive?

Generate IMPROVED primitives JSON. Use 15-30 primitives with better proportions and more distinctive features.
Respond with ONLY the JSON object, nothing else."""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 3000}
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json().get("response", "")
        
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            primitives = parsed.get("primitives", [])
            if primitives and len(primitives) > 0:
                print(f"[+] LLM generated {len(primitives)} REFINED primitives for '{object_name}'")
                valid = []
                for p in primitives:
                    if 'pos' in p and 'size' in p and len(p['pos']) == 3 and len(p['size']) == 3:
                        if 'color' not in p:
                            p['color'] = '#888888'
                        if 'name' not in p:
                            p['name'] = 'block'
                        valid.append(p)
                if valid:
                    return valid
        
        print(f"[!] LLM refinement response could not be parsed")
    except Exception as e:
        print(f"[!] LLM refinement failed: {e}")
    
    # If refinement fails, return original
    return previous_primitives


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", help="Name of the object to build")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-fp16")
    parser.add_argument("--grid-size", type=int, default=48)
    args = parser.parse_args()
    
    primitives = query_llm_for_primitives(args.object_name, model=args.model)
    print(f"\nPrimitives:")
    print(json.dumps(primitives, indent=2))
    
    grid, color_grid = build_voxel_grid_from_primitives(primitives, args.grid_size)

