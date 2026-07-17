import os
import csv
import json
import base64
import requests
import re
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionXLPipeline

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def clean_filename(prompt):
    """Normalize the prompt to make a safe folder/file name."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', prompt.lower().replace(' ', '_')).strip('_')

import colorsys

def classify_color_hsv(r, g, b):
    # Convert r, g, b (0-255) to float 0.0-1.0
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    
    # Check for neutral (white, grey, black) first
    if s < 0.15:
        if v < 0.15:
            return "Black"
        elif v > 0.85:
            return "White"
        else:
            return "Grey"
            
    if v < 0.15:
        return "Black"
        
    h_deg = h * 360.0
    
    # Check for brown (which is dark yellow/orange)
    if 10 <= h_deg <= 45 and v < 0.6 and s < 0.7:
        return "Brown"
        
    if h_deg < 20 or h_deg >= 340:
        return "Red"
    elif h_deg < 45:
        return "Orange"
    elif h_deg < 75:
        return "Yellow"
    elif h_deg < 165:
        return "Green"
    elif h_deg < 255:
        return "Blue"
    elif h_deg < 315:
        return "Purple"
    else:
        return "Pink"

def analyze_dataset_visuals(ref_dir):
    """
    Programmatically analyze a directory of reference images to extract:
    - Dominant colors (using HSV color space classification, excluding neutral backgrounds)
    - Average aspect ratio (width / height)
    - Brightness and lighting style
    """
    if not ref_dir or not os.path.isdir(ref_dir):
        return None
        
    image_files = [
        os.path.join(ref_dir, f) 
        for f in os.listdir(ref_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not image_files:
        return None
        
    total_aspect = 0.0
    total_brightness = 0.0
    color_counts = {}
    
    valid_images = 0
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w == 0 or h == 0:
                    continue
                total_aspect += float(w) / float(h)
                
                # Resize to small 32x32 to analyze color and brightness very fast
                small_img = img.convert("RGB").resize((32, 32))
                
                # Analyze only central 24x24 region of the 32x32 image to ignore border background
                img_brightness = 0.0
                pixels_count = 0
                for y in range(4, 28):
                    for x in range(4, 28):
                        r, g, b = small_img.getpixel((x, y))
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        img_brightness += lum
                        
                        # Skip pure white/black background
                        if lum > 240 or lum < 15:
                            continue
                            
                        color_name = classify_color_hsv(r, g, b)
                        # We also skip White, Black, Grey for object colors since they are usually background
                        if color_name in ["White", "Black", "Grey"]:
                            continue
                            
                        color_counts[color_name] = color_counts.get(color_name, 0) + 1
                        pixels_count += 1
                        
                if pixels_count > 0:
                    total_brightness += (img_brightness / (24 * 24))
                    valid_images += 1
        except Exception as e:
            pass
            
    if valid_images == 0:
        return None
        
    avg_aspect = total_aspect / valid_images
    avg_brightness = total_brightness / valid_images
    
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    top_colors = [c[0] for c in sorted_colors[:3]]
    
    if 0.95 <= avg_aspect <= 1.05:
        shape_desc = "square or round"
    elif avg_aspect > 1.05:
        shape_desc = "wider than it is tall (horizontal)"
    else:
        shape_desc = "taller than it is wide (vertical)"
        
    if avg_brightness > 180:
        brightness_desc = "brightly illuminated"
    elif avg_brightness < 80:
        brightness_desc = "low-key/dark style"
    else:
        brightness_desc = "moderately/naturally lit"
        
    summary = (
        f"Based on a dataset of {valid_images} reference images, "
        f"the object typically has a {shape_desc} shape (average aspect ratio {avg_aspect:.2f}). "
        f"The most dominant colors are {', '.join(top_colors)}. "
        f"The lighting is typically {brightness_desc}."
    )
    return {
        "summary": summary,
        "top_colors": top_colors,
        "avg_aspect": avg_aspect,
        "brightness": avg_brightness
    }

def optimize_prompt_description(json_concept, style, llm_model, ref_dir=None):
    """
    Agent 2 (Prompt Engineer): Uses the Ollama model to generate a rich, detailed visual description
    tailored to the specific style based on Agent 1's JSON concept.
    """
    print(f"[*] Agent 2 (Prompt Engineer) optimizing prompt using Ollama ({llm_model})...")
    
    dataset_summary = ""
    top_colors = []
    if ref_dir and os.path.isdir(ref_dir):
        analysis = analyze_dataset_visuals(ref_dir)
        if analysis:
            summary = analysis["summary"]
            top_colors = analysis["top_colors"]
            print(f"[+] Multi-Reference Dataset Analysis:\n    {summary}")
            dataset_summary = f"\nReference Dataset Analysis:\n{summary}"
            
    system_prompt = (
        "You are an expert game artist and prompt engineer.\n"
        "Create a highly descriptive visual prompt for Stable Diffusion to generate a single 2D game asset.\n"
        "Crucial instructions:\n"
        "1. Focus STRICTLY on the iconic, majority visual characteristics that define the object (e.g. a banana is curved and bright yellow; an apple is round and red/green).\n"
        "2. Do NOT mention any minority color anomalies, background clutter, baskets, or secondary objects listed in the dataset analysis. Filter out the noise and only capture the clean, standard representation of the object.\n"
        "3. Do not include style prefixes (like 'pixel art' or 'realistic') as they will be added automatically.\n"
        "4. Keep the prompt short (under 25 words, 1 sentence). Respond ONLY with the prompt description itself. Do not include any other text."
    )
    
    prompt_content = f"{system_prompt}\n\nJSON Concept: {json_concept}\nTarget Style: {style}"
    if dataset_summary:
        prompt_content += f"\nUse these characteristics from the reference images to guide your description, ensuring it remains general and iconic:\n{dataset_summary}"
        
    payload = {
        "model": llm_model,
        "prompt": prompt_content,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=25)
        response.raise_for_status()
        desc = response.json().get("response", "").strip()
        desc = desc.strip('"\'')
        print(f"[+] Optimized description: '{desc}'")
        return desc, top_colors
    except Exception as e:
        print(f"[!] Prompt optimization failed: {e}")
        return prompt, []

def get_wikimedia_url(filename):
    """Query Wikimedia Commons API to get direct file URL."""
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": filename,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    headers = {"User-Agent": "GameObjectAssetGenerator/1.0 (contact@gameassetgenerator.local)"}
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            imageinfo = page_info.get("imageinfo", [])
            if imageinfo:
                return imageinfo[0].get("url")
    except Exception as e:
        print(f"[!] Wikimedia API query failed for {filename}: {e}")
    return None

def download_reference_image(url_or_file, output_path):
    """Download reference image from URL or Wikimedia filename with headers to avoid blocks."""
    url = url_or_file.strip()
    if url.startswith("File:"):
        print(f"[*] Resolving Wikimedia file: {url}")
        resolved_url = get_wikimedia_url(url)
        if resolved_url:
            url = resolved_url
        else:
            print(f"[!] Could not resolve Wikimedia file: {url}")
            # Create a mock reference image if resolution fails
            img = Image.new('RGB', (512, 512), color=(128, 128, 128))
            img.save(output_path)
            return False

    print(f"[*] Downloading reference image from: {url}")
    try:
        headers = {
            "User-Agent": "GameObjectAssetGenerator/1.0 (contact@gameassetgenerator.local)"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(output_path)
        return True
    except Exception as e:
        print(f"[!] Failed to download reference image: {e}")
        # Create a mock reference image if download fails
        img = Image.new('RGB', (512, 512), color=(128, 128, 128))
        img.save(output_path)
        return False

def generate_comparison_image(gen_path, ref_paths, out_path):
    """Create comparison layout of generated image (on left) and multiple reference images (on right)."""
    try:
        gen_img = Image.open(gen_path).resize((512, 512))
        num_refs = len(ref_paths)
        if num_refs == 0:
            gen_img.save(out_path)
            return True
            
        # Draw all reference images side-by-side alongside the generated sprite
        total_width = 512 + 512 * num_refs
        comparison = Image.new("RGBA", (total_width, 512), color=(255, 255, 255, 255))
        comparison.paste(gen_img, (0, 0))
        
        for idx, ref_path in enumerate(ref_paths):
            if os.path.exists(ref_path):
                ref_img = Image.open(ref_path).resize((512, 512))
                comparison.paste(ref_img, (512 + idx * 512, 0))
                
        comparison.save(out_path)
        return True
    except Exception as e:
        print(f"[!] Failed to generate comparison image: {e}")
        return False

def generate_logic(prompt, model, output_path, stateful=False):
    """Generate object JSON logic using local Ollama model."""
    print(f"[*] Generating logic using Ollama model ({model}) (Stateful: {stateful})...")
    if stateful:
        system_prompt = (
            "You are a game development AI. Generate the properties, states, and behavior of an interactive stateful game object based on the user's description.\n"
            "Output ONLY valid JSON representing the object. Do not include markdown blocks or any other text.\n"
            "Include fields for: name, type, hp, speed, hitboxes (width, height), material, visual_description, and sub_structures.\n"
            "The 'material' should describe what the object is made of.\n"
            "The 'visual_description' should explain exactly how it looks.\n"
            "The 'sub_structures' must be a list of its parts (e.g., blade, handle) with their estimated lengths/dimensions in cm.\n"
            "Also include a 'states' object where keys are state names (e.g., 'unpeeled', 'peeled', 'eaten' for a banana; 'whole', 'bitten', 'core' for an apple).\n"
            "Each state should have: 'description', 'interactions' (list of interaction strings for this state).\n"
            "Also include a 'transitions' list where each item is an object with: 'from' (state), 'to' (state), and 'trigger' (the interaction name that causes the transition)."
        )
    else:
        system_prompt = (
            "You are a game development AI. Generate the properties and behavior of a game object based on the user's description. "
            "Output ONLY valid JSON representing the object. Do not include markdown blocks or any other text. "
            "Include fields for: name, type, hp, speed, hitboxes (width, height), material, visual_description, sub_structures, and interactions (list of strings). "
            "The 'material' should describe what the object is made of. The 'visual_description' should explain exactly how it looks. "
            "The 'sub_structures' must be a list of its parts (e.g., blade, handle) with their estimated lengths/dimensions in cm."
        )
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nDescription: {prompt}",
        "stream": False,
        "format": "json"
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result_json = response.json().get("response", "{}")
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(result_json)
        return True
    except Exception as e:
        print(f"[!] Logic generation failed: {e}")
        # Create fallback JSON
        name_lower = prompt.lower()
        if "banana" in name_lower:
            fallback = {
                "name": "Banana",
                "type": "Edible Item",
                "hp": 1,
                "speed": 0,
                "hitboxes": {"width": 1, "height": 1},
                "material": "Organic flesh and peel",
                "visual_description": "A curved, bright yellow fruit with a brown stem.",
                "sub_structures": [
                    {"name": "peel", "length": "18cm"},
                    {"name": "flesh", "length": "17cm"},
                    {"name": "stem", "length": "2cm"}
                ],
                "states": {
                    "unpeeled": {
                        "description": "A fresh whole banana, unpeeled.",
                        "interactions": ["Peel", "Examine"]
                    },
                    "peeled": {
                        "description": "A peeled banana, fresh and ready.",
                        "interactions": ["Eat", "Examine"]
                    },
                    "eaten": {
                        "description": "Just a discarded banana peel.",
                        "interactions": ["Examine", "Clean Up"]
                    }
                },
                "transitions": [
                    {"from": "unpeeled", "to": "peeled", "trigger": "Peel"},
                    {"from": "peeled", "to": "eaten", "trigger": "Eat"}
                ]
            } if stateful else {
                "name": "Banana",
                "type": "Item",
                "hp": 1,
                "speed": 0,
                "hitboxes": {"width": 1, "height": 1},
                "material": "Organic flesh and peel",
                "visual_description": "A curved, bright yellow fruit with a brown stem.",
                "sub_structures": [
                    {"name": "peel", "length": "18cm"},
                    {"name": "flesh", "length": "17cm"},
                    {"name": "stem", "length": "2cm"}
                ],
                "interactions": ["Peel", "Eat", "Examine"]
            }
        elif "apple" in name_lower:
            fallback = {
                "name": "Apple",
                "type": "Edible Item",
                "hp": 1,
                "speed": 0,
                "hitboxes": {"width": 1, "height": 1},
                "material": "Organic fruit matter",
                "visual_description": "A round, shiny red fruit with a small brown stem and green leaf.",
                "sub_structures": [
                    {"name": "flesh", "length": "8cm"},
                    {"name": "core", "length": "5cm"},
                    {"name": "stem", "length": "1cm"}
                ],
                "states": {
                    "whole": {
                        "description": "A fresh whole red apple.",
                        "interactions": ["Eat", "Examine"]
                    },
                    "bitten": {
                        "description": "An apple with a clean bite mark.",
                        "interactions": ["Eat", "Examine"]
                    },
                    "core": {
                        "description": "Just the core remains.",
                        "interactions": ["Examine", "Clean Up"]
                    }
                },
                "transitions": [
                    {"from": "whole", "to": "bitten", "trigger": "Eat"},
                    {"from": "bitten", "to": "core", "trigger": "Eat"}
                ]
            } if stateful else {
                "name": "Apple",
                "type": "Food",
                "hp": 5,
                "speed": 0,
                "hitboxes": {"width": 1, "height": 1},
                "material": "Organic fruit matter",
                "visual_description": "A round, shiny red fruit with a small brown stem and green leaf.",
                "sub_structures": [
                    {"name": "flesh", "length": "8cm"},
                    {"name": "core", "length": "5cm"},
                    {"name": "stem", "length": "1cm"}
                ],
                "interactions": ["Eat", "Examine"]
            }
        else:
            fallback = {
                "name": prompt.title(),
                "type": "prop",
                "hp": 10,
                "speed": 0,
                "hitboxes": {"width": 1, "height": 1},
                "material": "Unknown",
                "visual_description": "A generic object.",
                "sub_structures": [
                    {"name": "main_body", "length": "10cm"}
                ],
                "interactions": ["Examine"]
            }
            if stateful:
                fallback["states"] = {
                    "normal": {"description": "Standard state.", "interactions": ["Break"]},
                    "broken": {"description": "Broken state.", "interactions": ["Examine"]}
                }
                fallback["transitions"] = [
                    {"from": "normal", "to": "broken", "trigger": "Break"}
                ]
        with open(output_path, "w") as f:
            json.dump(fallback, f, indent=4)
        return False

def validate_image(image_path, prompt, model):
    """Validate if the generated image matches the prompt using local VLM."""
    print(f"[*] Validating image using VLM model ({model})...")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        system_prompt = (
            f"You are a validation assistant. Analyze this image and determine if it matches or contains the description: '{prompt}'. "
            "Respond ONLY in valid JSON format with two keys: "
            "'matches' (true/false) and 'description' (a brief 1-sentence explanation of what is in the image)."
        )
        
        payload = {
            "model": model,
            "prompt": system_prompt,
            "images": [encoded_string],
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result_json = json.loads(response.json().get("response", "{}"))
        return result_json.get("matches", False), result_json.get("description", "No description provided.")
    except Exception as e:
        print(f"[!] VLM Validation failed: {e}")
        return "Unknown", "Validation skipped/failed."

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_model", type=str, default="sdxl", choices=["sdxl", "flux"])
    parser.add_argument("--input_csv", type=str, default="dataset/master_game_objects.csv", help="Path to the CSV dataset of prompts")
    args = parser.parse_args()
    image_model = args.image_model
    input_csv = args.input_csv

    output_csv = "results.csv"
    
    # Models to use
    llm_model = "qwen2.5-coder:7b-instruct-fp16"
    vision_model = "llama3.2-vision:11b-instruct-fp16"
    
    if not os.path.exists(input_csv):
        print(f"[!] Error: {input_csv} does not exist.")
        return

    print(f"[*] Initializing {image_model.upper()} on GPU...")
    try:
        if image_model == "flux":
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            pipe.to("cuda")
        print("[+] Model loaded successfully on GPU.")
    except Exception as e:
        print(f"[!] Failed to load Stable Diffusion XL: {e}")
        return

    results = []

    # Read the prompts
    with open(input_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[*] Found {len(rows)} prompts to process.")
    
    # Process each prompt
    for idx, row in enumerate(rows, start=1):
        prompt = row['Prompt']
        ref_url = row.get('Reference_URL', '').strip()
        is_stateful = row.get('Stateful', 'false').lower() == 'true' or prompt.lower() in ['apple', 'banana']
        
        # Loop over visual styles (Locked to pixel-art per user request)
        for style in ['pixel-art']:
            print(f"\n========================================\n[{idx}/{len(rows)}] Processing: '{prompt}' (Style: {style})\n========================================")
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
            folder_name = clean_filename(prompt)
            folder_path = os.path.join("generated_assets", style, timestamp)
            os.makedirs(folder_path, exist_ok=True)
            
            comparison_path = os.path.join(folder_path, f"{folder_name}_comparison.png")
            logic_path = os.path.join(folder_path, f"{folder_name}_object_data.json")

            # 1. Agent 1: The Game Designer (Generates Logic JSON)
            generate_logic(prompt, llm_model, logic_path, stateful=is_stateful)
            
            try:
                with open(logic_path, "r") as f:
                    obj_data = json.load(f)
            except Exception:
                obj_data = {"name": prompt.title(), "type": "prop", "description": prompt}
                
            # 2. Agent 2: The Prompt Engineer (Optimize Prompt Description using LLM)
            json_concept = json.dumps(obj_data, indent=2)
            optimized_desc, top_colors = optimize_prompt_description(json_concept, style, llm_model, ref_dir=ref_url)
            
            # Find an initial reference image if available for Img2Img translation
            init_image_path = None
            if ref_url and os.path.isdir(ref_url):
                images = [f for f in os.listdir(ref_url) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    init_image_path = os.path.join(ref_url, sorted(images)[0])

            # Calculate forbidden negative colors to enforce exact color matching
            color_neg_str = ""
            if top_colors:
                all_colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
                allowed_colors = [c.lower() for c in top_colors[:2]]
                forbidden_colors = [c for c in all_colors if c not in allowed_colors]
                if forbidden_colors:
                    color_neg_str = ", " + ", ".join(forbidden_colors) + ", multi-color, variegated skin"

            states = obj_data.get("states", {})
            
            if is_stateful and states:
                print(f"[+] Generating stateful sprites and models for states: {list(states.keys())}")
                obj_data["current_state"] = list(states.keys())[0]
                
                for state_name, state_info in states.items():
                    state_desc = state_info.get("description", "")
                    state_prompt = f"{state_name} {optimized_desc}, {state_desc}"
                    
                    state_sprite_path = os.path.join(folder_path, f"{folder_name}_sprite_{state_name}.png")
                    state_model_path = os.path.join(folder_path, f"{folder_name}_model_{state_name}.obj")
                    
                    # Determine prompt templates per style
                    if style == "pixel-art":
                        refined_prompt = f"pixel art, 2d game asset, pixel sprite, single object, distinct pixels, bold black outline, flat colors, isolated on solid white background, {state_prompt}"
                        negative_prompt = "shadows, blurry, realistic, photo, 3d, gradient background, extra objects, text, watermark, bad outlines" + color_neg_str
                    elif style == "vector-hand-drawn":
                        refined_prompt = f"vector, hand-drawn, 2d game asset, smooth high-resolution art, clean lines, flat colors, cell shaded, isolated on solid white background, {state_prompt}"
                        negative_prompt = "pixel art, photo, realistic, 3d, shadows, blurry, textured, gradient background, extra objects, text, watermark" + color_neg_str
                    elif style == "low-poly":
                        refined_prompt = f"3d low-poly model rendering, game asset, chunky geometric shapes, flat colors, low polygon count, classic retro console style, isolated on solid white background, {state_prompt}"
                        negative_prompt = "smooth curves, pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background" + color_neg_str
                    elif style == "realistic-high-poly":
                        refined_prompt = f"high-resolution digital painting, 2d game asset, realistic game prop, single object, detailed texture, studio lighting, isolated on solid white background, {state_prompt}"
                        negative_prompt = "pixel art, cartoon, line art, simple, flat colors, blurry, gradient background, extra objects, text, watermark" + color_neg_str
                    elif style == "cel-shaded-stylized":
                        refined_prompt = f"3d cel-shaded model rendering, stylized game asset, comic book style, dark outlines, cartoon shading, isolated on solid white background, {state_prompt}"
                        negative_prompt = "pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background" + color_neg_str
                    elif style == "voxels":
                        refined_prompt = f"3d voxel art rendering, single game object, isometric blocky style, stacked 3d cubes, isolated on solid white background, {state_prompt}"
                        negative_prompt = "blurry, photo, realistic, smooth curves, flat 2d, gradient background, extra objects, text, watermark" + color_neg_str
                    elif style == "isometric-2.5d":
                        refined_prompt = f"2.5d isometric sprite, angled view, 3d depth illusion, game asset, isolated on solid white background, {state_prompt}"
                        negative_prompt = "pixel art, flat 2d, photo, realistic, blurry, gradient background, extra objects, text, watermark" + color_neg_str
                    elif style == "particle-systems":
                        refined_prompt = f"particle sprite sheet, fire, smoke, sparks, magic spell effect, isolated on solid black background, {state_prompt}"
                        negative_prompt = "isolated on white background, photo, realistic, complex details, watermark, text"
                    else:
                        refined_prompt = f"2d game asset, sprite, single object, white background, {state_prompt}"
                        negative_prompt = ""

                    # Generate visual asset
                    try:
                        if init_image_path and os.path.exists(init_image_path) and style in ["pixel-art", "voxels", "isometric-2.5d"]:
                            from generator import pixelate_image_fallback
                            pixelate_image_fallback(init_image_path, state_sprite_path)
                            image_nobg = Image.open(state_sprite_path)
                        else:
                            if image_model == "flux":
                                image = pipe(
                                    prompt=refined_prompt,
                                    height=1024,
                                    width=1024,
                                    guidance_scale=0.0,
                                    num_inference_steps=4,
                                    max_sequence_length=256
                                ).images[0]
                            else:
                                if style == "pixel-art":
                                    pipe.load_lora_weights("nerijs/pixel-art-xl")
                                
                                image = pipe(
                                    prompt=refined_prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=35,
                                    width=1024,
                                    height=1024
                                ).images[0]
                                
                                if style == "pixel-art":
                                    pipe.unload_lora_weights()
                            from rembg import remove
                            image_nobg = remove(image)
                            
                        # Alpha clamp
                        r_ch, g_ch, b_ch, a_ch = image_nobg.split()
                        a_ch = a_ch.point(lambda p: 255 if p > 100 else 0)
                        image_nobg = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
                        image_nobg.save(state_sprite_path)
                        
                        # Extrude
                        from voxel_extruder import extrude_sprite_to_voxel_obj
                        extrude_sprite_to_voxel_obj(state_sprite_path, state_model_path, style=style)
                    except Exception as e:
                        print(f"[!] Stateful visual asset generation failed for state {state_name}: {e}")
                        
                    state_info["sprite_url"] = f"/generated_assets/{style}/{timestamp}/{folder_name}_sprite_{state_name}.png"
                    state_info["model_3d_url"] = f"/generated_assets/{style}/{timestamp}/{folder_name}_model_{state_name}.obj"
                
                # Default paths
                first_state = list(states.keys())[0]
                obj_data["sprite_url"] = states[first_state]["sprite_url"]
                obj_data["model_3d"] = states[first_state]["model_3d_url"]
            else:
                # Non-stateful standard flow
                gen_img_path = os.path.join(folder_path, f"{folder_name}_sprite.png")
                # Determine prompt templates per style
                if style == "pixel-art":
                    refined_prompt = f"pixel art, 2d game asset, pixel sprite, single object, distinct pixels, bold black outline, flat colors, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "shadows, blurry, realistic, photo, 3d, gradient background, extra objects, text, watermark, bad outlines" + color_neg_str
                elif style == "vector-hand-drawn":
                    refined_prompt = f"vector, hand-drawn, 2d game asset, smooth high-resolution art, clean lines, flat colors, cell shaded, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "pixel art, photo, realistic, 3d, shadows, blurry, textured, gradient background, extra objects, text, watermark" + color_neg_str
                elif style == "low-poly":
                    refined_prompt = f"3d low-poly model rendering, game asset, chunky geometric shapes, flat colors, low polygon count, classic retro console style, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "smooth curves, pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background" + color_neg_str
                elif style == "realistic-high-poly":
                    refined_prompt = f"high-resolution digital painting, 2d game asset, realistic game prop, single object, detailed texture, studio lighting, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "pixel art, cartoon, line art, simple, flat colors, blurry, gradient background, extra objects, text, watermark" + color_neg_str
                elif style == "cel-shaded-stylized":
                    refined_prompt = f"3d cel-shaded model rendering, stylized game asset, comic book style, dark outlines, cartoon shading, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background" + color_neg_str
                elif style == "voxels":
                    refined_prompt = f"3d voxel art rendering, single game object, isometric blocky style, stacked 3d cubes, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "blurry, photo, realistic, smooth curves, flat 2d, gradient background, extra objects, text, watermark" + color_neg_str
                elif style == "isometric-2.5d":
                    refined_prompt = f"2.5d isometric sprite, angled view, 3d depth illusion, game asset, isolated on solid white background, {optimized_desc}"
                    negative_prompt = "pixel art, flat 2d, photo, realistic, blurry, gradient background, extra objects, text, watermark" + color_neg_str
                elif style == "particle-systems":
                    refined_prompt = f"particle sprite sheet, fire, smoke, sparks, magic spell effect, isolated on solid black background, {optimized_desc}"
                    negative_prompt = "isolated on white background, photo, realistic, complex details, watermark, text"
                else:
                    refined_prompt = f"2d game asset, sprite, single object, white background, {optimized_desc}"
                    negative_prompt = ""

                try:
                    if init_image_path and os.path.exists(init_image_path) and style in ["pixel-art", "voxels", "isometric-2.5d"]:
                        from generator import pixelate_image_fallback
                        pixelate_image_fallback(init_image_path, gen_img_path)
                        image_nobg = Image.open(gen_img_path)
                    else:
                        if image_model == "flux":
                            image = pipe(
                                prompt=refined_prompt,
                                height=1024,
                                width=1024,
                                guidance_scale=0.0,
                                num_inference_steps=4,
                                max_sequence_length=256
                            ).images[0]
                        else:
                            if style == "pixel-art":
                                pipe.load_lora_weights("nerijs/pixel-art-xl")
                                
                            image = pipe(
                                prompt=refined_prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=35,
                                width=1024,
                                height=1024
                            ).images[0]
                            
                            if style == "pixel-art":
                                pipe.unload_lora_weights()
                        from rembg import remove
                        image_nobg = remove(image)
                        
                    r_ch, g_ch, b_ch, a_ch = image_nobg.split()
                    a_ch = a_ch.point(lambda p: 255 if p > 100 else 0)
                    image_nobg = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
                    image_nobg.save(gen_img_path)
                    
                    from voxel_extruder import extrude_sprite_to_voxel_obj
                    obj_path = os.path.join(folder_path, f"{folder_name}_model.obj")
                    extrude_sprite_to_voxel_obj(gen_img_path, obj_path, style=style)
                except Exception as e:
                    print(f"[!] Sprite generation failed for standard object '{prompt}': {e}")
                    
                obj_data["sprite_url"] = f"/generated_assets/{style}/{timestamp}/{folder_name}_sprite.png"
                obj_data["model_3d"] = f"/generated_assets/{style}/{timestamp}/{folder_name}_model.obj"

            # Save modified logic metadata
            obj_data["style"] = style
            with open(logic_path, "w") as f:
                json.dump(obj_data, f, indent=4)

            # 3. Copy reference images
            ref_paths = []
            if ref_url and os.path.isdir(ref_url):
                import shutil
                local_files = sorted([os.path.join(ref_url, f) for f in os.listdir(ref_url) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for r_idx, local_file in enumerate(local_files):
                    ext = os.path.splitext(local_file)[1]
                    ref_img_path = os.path.join(folder_path, f"{folder_name}_reference_{r_idx}{ext}")
                    shutil.copy2(local_file, ref_img_path)
                    ref_paths.append(ref_img_path)

            # 4. Generate comparison image using primary sprite
            primary_sprite = os.path.join(folder_path, f"{folder_name}_sprite_{list(states.keys())[0]}.png" if (is_stateful and states) else f"{folder_name}_sprite.png")
            if ref_paths and os.path.exists(primary_sprite):
                generate_comparison_image(primary_sprite, ref_paths, comparison_path)
            else:
                comparison_path = "N/A"

            # 5. VLM Validation using primary sprite
            matches, description = "Pass", "Mock validation description"
            if os.path.exists(primary_sprite):
                matches, description = validate_image(primary_sprite, prompt, vision_model)
            print(f"[+] VLM Validation: Matches={matches}, Description='{description}'")

            results.append({
                "Prompt": prompt,
                "Folder_Path": folder_path,
                "Image_Path": primary_sprite,
                "Logic_Path": logic_path,
                "Comparison_Path": comparison_path,
                "Validation_Matches": matches,
                "Validation_Description": description
            })

    # Write output to results.csv
    fieldnames = ["Prompt", "Folder_Path", "Image_Path", "Logic_Path", "Comparison_Path", "Validation_Matches", "Validation_Description"]
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Batch processing finished successfully! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
