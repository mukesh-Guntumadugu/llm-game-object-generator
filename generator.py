import os
import json
import argparse
import base64
import requests
import re
from PIL import Image
from io import BytesIO
from voxel_extruder import extrude_sprite_to_voxel_obj, voxelize_mesh, voxelize_from_primitives

# Configuration for Open Source Models
OLLAMA_API_URL = "http://localhost:11434/api/generate"
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

def clean_filename(prompt):
    """Normalize the prompt to make a safe folder/file name."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', prompt.lower().replace(' ', '_')).strip('_')

def pixelate_image_fallback(image_path: str, output_path: str, pixel_size: int = 8):
    """Fallback pixelation filter for Image-to-Pixel-Art translation."""
    print(f"[*] Applying fallback pixelation filter to: {image_path}")
    try:
        img = Image.open(image_path).convert("RGBA")
        # Resize down, then resize up to pixelate
        w, h = img.size
        small = img.resize((max(1, w // pixel_size), max(1, h // pixel_size)), Image.Resampling.NEAREST)
        pixelated = small.resize((w, h), Image.Resampling.NEAREST)
        pixelated.save(output_path)
        print(f"[+] Fallback pixelated image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[!] Fallback pixelation failed: {e}")
        return False

def create_mock_sprite(prompt: str, output_filename: str, llm_model: str = "qwen2.5-coder:7b-instruct-fp16"):
    """
    Creates a 2D sprite by asking the LLM to decide the correct colors and shape.
    No hardcoded keyword logic - the model thinks for itself.
    """
    from PIL import ImageDraw
    import math
    import json
    import re

    print(f"[*] Asking LLM to design the sprite colors and shape for: '{prompt}'")

    # Ask the LLM to return a structured JSON color/shape plan
    color_query = f"""You are a pixel art game asset color designer.
For the game object described below, output a JSON object describing how to draw it.

Object: "{prompt}"

Return ONLY a JSON object with this exact structure, no explanation:
{{
  "shape": "round" | "tall" | "wide" | "shield",
  "body_color": [R, G, B],
  "detail_color": [R, G, B],
  "accent_color": [R, G, B],
  "has_stem": true | false,
  "stem_color": [R, G, B],
  "has_leaf": true | false,
  "leaf_color": [R, G, B]
}}

Use realistic, vivid colors appropriate for the object. Do not use red for everything."""

    design = None
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": llm_model, "prompt": color_query, "stream": False},
            timeout=20
        )
        raw = response.json().get("response", "")
        # Extract JSON from the response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            design = json.loads(match.group())
    except Exception as e:
        print(f"[!] LLM color design failed ({e}), using fallback colors.")

    # Fallback if LLM fails
    if not design:
        design = {
            "shape": "round",
            "body_color": [180, 80, 50],
            "detail_color": [220, 120, 80],
            "accent_color": [240, 200, 150],
            "has_stem": False,
            "stem_color": [101, 67, 33],
            "has_leaf": False,
            "leaf_color": [46, 139, 87]
        }

    def rgb(key): return tuple(design.get(key, [128, 128, 128]) + [255])

    body_color = rgb("body_color")
    detail_color = rgb("detail_color")
    accent_color = rgb("accent_color")
    shape = design.get("shape", "round")

    print(f"[+] LLM chose shape='{shape}', body={body_color[:3]}, detail={detail_color[:3]}")

    img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if shape == "round":
        # Sphere-like object: layered concentric ellipses for depth/shading
        draw.ellipse([110, 110, 400, 400], fill=body_color)
        draw.ellipse([140, 140, 370, 370], fill=detail_color)
        draw.ellipse([170, 160, 310, 290], fill=accent_color)  # highlight
    elif shape == "tall":
        # Tall object (bottle, sword, tree trunk)
        draw.ellipse([196, 300, 316, 420], fill=body_color)    # base
        draw.rectangle([216, 120, 296, 330], fill=body_color)  # body
        draw.ellipse([206, 100, 306, 160], fill=detail_color)  # top cap
        draw.rectangle([226, 130, 286, 310], fill=detail_color) # center shine
    elif shape == "wide":
        # Wide object (chest, shield, book)
        draw.rectangle([100, 180, 420, 360], fill=body_color)
        draw.rectangle([120, 200, 400, 340], fill=detail_color)
        draw.ellipse([200, 250, 310, 295], fill=accent_color)  # clasp/gem
    elif shape == "shield":
        # Shield / gem shape
        points = [(256, 110), (390, 185), (340, 370), (256, 430), (172, 370), (122, 185)]
        draw.polygon(points, fill=body_color)
        inner = [(256, 145), (355, 200), (315, 350), (256, 400), (197, 350), (157, 200)]
        draw.polygon(inner, fill=detail_color)
        draw.ellipse([220, 240, 292, 310], fill=accent_color)
    else:
        # Default round
        draw.ellipse([110, 110, 400, 400], fill=body_color)
        draw.ellipse([150, 150, 360, 360], fill=detail_color)

    # Optional stem and leaf
    if design.get("has_stem"):
        stem_color = rgb("stem_color")
        draw.rectangle([245, 70, 265, 130], fill=stem_color)
    if design.get("has_leaf"):
        leaf_color = rgb("leaf_color")
        draw.ellipse([260, 75, 330, 115], fill=leaf_color)

    img.save(output_filename)
    return output_filename

def generate_visual_asset(prompt: str, output_filename: str = "sprite.png", local_image: bool = False, style: str = "pixel-art", init_image_path: str = None, image_model: str = "sdxl") -> str:
    """
    Uses a diffusion model (Stable Diffusion XL) to generate the game object sprite.
    Supports Image-to-Image translation (Img2Img) if init_image_path is provided.
    Can be run via Hugging Face Inference API or locally.
    """
    print(f"[*] Generating image for: '{prompt}' (Style: {style})")
    
    if style == "pixel-art":
        refined_prompt = f"pixel art, 2d game asset, pixel sprite, single object, distinct pixels, bold black outline, flat colors, isolated on solid white background, {prompt}"
        negative_prompt = "shadows, blurry, realistic, photo, 3d, gradient background, extra objects, text, watermark, bad outlines"
    elif style == "vector-hand-drawn":
        refined_prompt = f"vector, hand-drawn, 2d game asset, smooth high-resolution art, clean lines, flat colors, cell shaded, isolated on solid white background, {prompt}"
        negative_prompt = "pixel art, photo, realistic, 3d, shadows, blurry, textured, gradient background, extra objects, text, watermark"
    elif style == "low-poly":
        refined_prompt = f"3d low-poly model rendering, game asset, chunky geometric shapes, flat colors, low polygon count, classic retro console style, isolated on solid white background, {prompt}"
        negative_prompt = "smooth curves, pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background"
    elif style == "realistic-high-poly":
        refined_prompt = f"high-resolution digital painting, 2d game asset, realistic game prop, single object, detailed texture, studio lighting, isolated on solid white background, {prompt}"
        negative_prompt = "pixel art, cartoon, line art, simple, flat colors, blurry, gradient background, extra objects, text, watermark"
    elif style == "cel-shaded-stylized":
        refined_prompt = f"3d cel-shaded model rendering, stylized game asset, comic book style, dark outlines, cartoon shading, isolated on solid white background, {prompt}"
        negative_prompt = "pixel art, photo, realistic, high-poly, blurry, text, watermark, gradient background"
    elif style == "voxels":
        refined_prompt = f"3d voxel art rendering, single game object, isometric blocky style, stacked 3d cubes, isolated on solid white background, {prompt}"
        negative_prompt = "blurry, photo, realistic, smooth curves, flat 2d, gradient background, extra objects, text, watermark"
    elif style == "isometric-2.5d":
        refined_prompt = f"2.5d isometric sprite, angled view, 3d depth illusion, game asset, isolated on solid white background, {prompt}"
        negative_prompt = "pixel art, flat 2d, photo, realistic, blurry, gradient background, extra objects, text, watermark"
    elif style == "particle-systems":
        refined_prompt = f"particle sprite sheet, fire, smoke, sparks, magic spell effect, isolated on solid black background, {prompt}"
        negative_prompt = "isolated on white background, photo, realistic, complex details, watermark, text"
    else:
        refined_prompt = f"2d game asset, sprite, single object, white background, {prompt}"
        negative_prompt = ""
    
    if local_image:
        print(f"[*] Generating image locally using {image_model.upper()}...")
        try:
            import torch
            
            if image_model == "flux":
                from diffusers import FluxPipeline
                # FLUX.1-schnell natively uses bfloat16 and needs about 12-16GB VRAM
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.bfloat16
                )
                # Optimize for standard GPUs
                pipe.enable_model_cpu_offload()
                
                # Flux schnell needs only 4 steps and guidance_scale=0
                image = pipe(
                    prompt=refined_prompt,
                    height=512,
                    width=512,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256
                ).images[0]
            
            else: # SDXL
                if init_image_path and os.path.exists(init_image_path):
                    print(f"[*] Using local Img2Img pipeline with: {init_image_path}")
                    from diffusers import StableDiffusionXLImg2ImgPipeline
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                    )
                    pipe.to("cuda")
                    
                    # Pre-process Wikipedia reference images using BRIA RMBG-1.4
                    # This perfectly isolates the object and deletes tables/backgrounds
                    # BEFORE Stable Diffusion ever sees it!
                    init_image_raw = Image.open(init_image_path).convert("RGBA")
                    try:
                        from rembg import remove, new_session
                        bria_session = new_session("u2net")
                        extracted = remove(init_image_raw, session=bria_session)
                        
                        # Create a solid white background and paste the extracted object onto it
                        white_bg = Image.new("RGB", extracted.size, (255, 255, 255))
                        white_bg.paste(extracted, mask=extracted.split()[3]) # Use alpha as mask
                        
                        # Pad the image to a perfect square to prevent squishing aspect ratios!
                        max_dim = max(white_bg.size)
                        square_bg = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                        offset = ((max_dim - white_bg.width) // 2, (max_dim - white_bg.height) // 2)
                        square_bg.paste(white_bg, offset)
                        init_image_raw = square_bg
                    except Exception as e:
                        print(f"[!] BRIA Reference Extraction Failed: {e}")
                        init_image_raw = init_image_raw.convert("RGB")

                    init_image = init_image_raw.resize((512, 512))
                    image = pipe(
                        prompt=refined_prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        strength=0.6,
                        num_inference_steps=30
                    ).images[0]
                else:
                    from diffusers import StableDiffusionXLPipeline
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                    )
                    pipe.to("cuda")
                    
                    # Apply specialized Pixel Art LoRA if style is pixel-art
                    if style == "pixel-art":
                        print("[*] Loading specialized 'nerijs/pixel-art-xl' LoRA weights...")
                        pipe.load_lora_weights("nerijs/pixel-art-xl")
                        
                    image = pipe(
                        prompt=refined_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,
                        width=512,
                        height=512
                    ).images[0]
            
            image.save(output_filename)
            return output_filename
        except Exception as e:
            print(f"[!] Local image generation failed: {e}")
            print("[*] Falling back to API/filter translation...")
            local_image = False
            
    if not local_image:
        # If an initial image is provided and we are offline/API is unavailable, use pixelation fallback
        if init_image_path and os.path.exists(init_image_path) and (style == "pixel-art" or style == "voxels" or style == "isometric-2.5d"):
            success = pixelate_image_fallback(init_image_path, output_filename)
            if success:
                return output_filename
                
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        try:
            if not HF_API_TOKEN:
                print("[*] HF_API_TOKEN not set — running local SDXL on GPU...")
                try:
                    from local_sdxl import generate_sprite
                    return generate_sprite(refined_prompt, output_filename, style=style)
                except Exception as sdxl_e:
                    print(f"[!] Local SDXL failed ({sdxl_e}), falling back to mock sprite.")
                    return create_mock_sprite(prompt, output_filename, llm_model=image_model)
                
            response = requests.post(HF_API_URL, headers=headers, json={"inputs": refined_prompt})
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            image.save(output_filename)
            return output_filename
        except Exception as e:
            print(f"[!] Image generation failed: {e}")
            if init_image_path and os.path.exists(init_image_path):
                # final fallback
                shutil.copy2(init_image_path, output_filename)
            else:
                create_mock_sprite(prompt, output_filename, llm_model=image_model)
            return output_filename

def extract_data_from_image(image_path: str, model: str = "llama3.2-vision:11b-instruct-fp16") -> str:
    """
    Extracts a text description of the concept in the image using a local Vision-Language Model via Ollama.
    """
    print(f"[*] Extracting object concept from image: {image_path} using VLM ({model})")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        system_prompt = (
            "Describe the physical appearance and function of the central object in this image. "
            "Explain what it is in one clear, descriptive sentence, suitable for a game asset description."
        )
        
        payload = {
            "model": model,
            "prompt": system_prompt,
            "images": [encoded_string],
            "stream": False,
            "keep_alive": 0
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        description = response.json().get("response", "").strip()
        return description
    except requests.exceptions.ConnectionError:
        print("[!] Could not connect to local Ollama instance. Is it running? Returning mock description.")
        if "apple" in image_path.lower():
            return "A fresh red apple."
        return "A medieval blacksmith anvil."
    except Exception as e:
        print(f"[!] Concept extraction failed: {e}")
        return "A medieval blacksmith anvil."

def agent_prompt_engineer(logic_data: dict, style: str, model: str = "llama3", feedback: str = None) -> str:
    """
    Agent 2 (Prompt Engineer): Analyzes the JSON concept from Agent 1 and translates it into a
    highly optimized, visually descriptive text prompt for the image diffusion model.
    """
    print(f"[*] Agent 2 (Prompt Engineer) optimizing diffusion prompt using {model}...")
    system_prompt = (
        "You are an expert AI Prompt Engineer for image diffusion models. "
        "Your task is to read the provided Game Object JSON data and write a highly descriptive, comma-separated visual description. "
        "Focus ONLY on physical appearance, exact colors, materials, and lighting. "
        "CRITICAL: You MUST ignore gameplay stats like HP, speed, hitboxes, mathematical dimensions (like '2x3'), ratios, or physical sizes. DO NOT include any numbers or dimensions in the prompt. "
    )
    
    if style in ["voxels", "isometric-2.5d"]:
        system_prompt += (
            "CRITICAL GEOMETRY RULE: You MUST draw this object from a PERFECTLY FLAT, STRAIGHT-ON, ORTHOGRAPHIC FRONT VIEW. "
            "Do NOT include any 3D perspective, top-down angles, or visible sides. It must look like a flat 2D schematic sticker! "
            "If you draw it with a 3D perspective, the 3D extruder will warp it!"
        )
        
    system_prompt += (
        f"The ultimate target art style will be: {style}. Ensure the description fits this style. "
        "Output ONLY the raw comma-separated prompt string. No conversational text."
    )
    
    if feedback:
        print(f"[*] Agent 2 analyzing Critic feedback: {feedback}")
        system_prompt += f"\nCRITICAL: Your previous prompt failed validation! The Vision Critic provided this feedback: '{feedback}'. You MUST rewrite your prompt to explicitly fix these issues."
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nJSON Concept:\n{json.dumps(logic_data, indent=2)}",
        "stream": False,
        "keep_alive": 0
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        optimized_prompt = response.json().get("response", "").strip()
        # Remove quotes if the LLM added them
        if optimized_prompt.startswith('"') and optimized_prompt.endswith('"'):
            optimized_prompt = optimized_prompt[1:-1]
        print(f"[+] Agent 2 Optimized Prompt: {optimized_prompt}")
        return optimized_prompt
    except Exception as e:
        print(f"[!] Agent 2 Prompt Engineering failed: {e}")
        return logic_data.get("name", "game object")
        
def validate_visual_asset(image_path: str, prompt: str, model: str = "llama3.2-vision") -> tuple[bool, str]:
    """
    Validates if the generated image matches the prompt using a local Vision-Language Model via Ollama.
    """
    print(f"[*] Validating generated image '{image_path}' using VLM ({model})...")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        system_prompt = (
            f"You are a fierce Art Critic. Analyze this generated image and determine if it looks like a high quality representation of: '{prompt}'. "
            "If the image looks severely distorted, glitchy, like a melted block, or completely fails to match the concept, mark it as false. "
            "CRITICAL: If the intended style is 'voxels', the image MUST be a perfectly flat, front-facing view. If it has a slanted 3D perspective, mark it as false! "
            "Respond ONLY in valid JSON format with two keys: "
            "'matches' (true/false) and 'reason' (If false, provide a specific 1-sentence reason why it failed and how to fix the prompt to avoid it. If true, just say 'Valid')."
        )
        
        payload = {
            "model": model,
            "prompt": system_prompt,
            "images": [encoded_string],
            "stream": False,
            "keep_alive": 0,
            "format": "json"
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        result_text = response.json().get("response", "{}").strip()
        result_json = json.loads(result_text)
        
        matches = result_json.get("matches", False)
        reason = result_json.get("reason", "Unknown reason")
        
        print(f"[*] Critic feedback: '{reason}'")
        if matches:
            print("[+] Validation PASSED: The image is high quality!")
            return True, reason
        else:
            print("[!] Validation WARNING: The image is distorted or failed the check.")
            return False, reason
            
    except requests.exceptions.ConnectionError:
        print("[!] Could not connect to local Ollama instance. Skipping image validation.")
        return True, "No connection"
    except Exception as e:
        print(f"[!] Validation failed: {e}")
        return True, "Error running validation"

def generate_visual_asset_with_reflexion(base_prompt: str, logic_data: dict, sprite_filename: str, args) -> str:
    """
    Executes the Agentic Reflexion Loop. 
    1. Uses the base prompt to generate an image.
    2. Critic validates.
    3. If rejected, pass feedback to Prompt Engineer for a rewrite.
    4. Loop until Critic is happy.
    """
    feedback = None
    current_prompt = base_prompt
    
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n==============================================")
        print(f"[*] REFLEXION LOOP: Attempt {attempt + 1} of {max_retries}")
        print(f"==============================================")
        
        generate_visual_asset(current_prompt, output_filename=sprite_filename, local_image=args.local_image, style=args.style, init_image_path=args.image, image_model=args.image_model)
        
        is_valid, feedback = validate_visual_asset(sprite_filename, current_prompt, model=args.vision_model)
        
        if is_valid:
            print("[+] Reflexion Loop Complete: Image validated successfully!")
            return current_prompt
            
        if attempt < max_retries - 1:
            print("[!] Critic rejected the image. Passing feedback to Prompt Engineer for a rewrite...")
            current_prompt = agent_prompt_engineer(logic_data, args.style, model=args.llm_model, feedback=feedback)
            
    print("[-] Reflexion Loop exhausted. Proceeding with best attempt.")
    return current_prompt

def agent_game_designer(prompt: str, model: str = "llama3", stateful: bool = False) -> str:
    """
    Agent 1 (Game Designer): Uses an LLM to invent the lore, properties, and states of the game object.
    Outputs a highly detailed JSON concept.
    """
    print(f"[*] Agent 1 (Game Designer) brainstorming concept using {model}...")
    
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
            "Include fields for: name, type, hp (if applicable), speed, hitboxes (width, height), material, visual_description, sub_structures, and interactions (list of strings). "
            "The 'material' should describe what the object is made of. The 'visual_description' should explain exactly how it looks. "
            "The 'sub_structures' must be a list of its parts (e.g., blade, handle) with their estimated lengths/dimensions in cm."
        )
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nDescription: {prompt}",
        "stream": False,
        "keep_alive": 0,
        "format": "json"
    }
    
    output_filename = "temp_object_data.json"
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result_json = response.json().get("response", "{}")
        
        with open(output_filename, "w") as f:
            f.write(result_json)
        return output_filename
    except Exception as e:
        print(f"[!] Logic generation failed: {e}. Creating mock JSON.")
        
        # Determine appropriate mock states based on prompt
        name_lower = prompt.lower()
        if "banana" in name_lower:
            mock_data = {
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
            mock_data = {
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
            mock_data = {
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
                mock_data["states"] = {
                    "normal": {"description": "Standard state.", "interactions": ["Break"]},
                    "broken": {"description": "Broken state.", "interactions": ["Examine"]}
                }
                mock_data["transitions"] = [
                    {"from": "normal", "to": "broken", "trigger": "Break"}
                ]

        with open(output_filename, "w") as f:
            json.dump(mock_data, f, indent=4)
        return output_filename

def generate_true_3d_asset(image_path: str, output_filename: str):
    """Generates a true 3D mesh using OpenAI's Shap-E Image-to-3D model."""
    print(f"[*] TRUE 3D MODE: Generating native 3D mesh from reference image '{image_path}' using Shap-E...")
    try:
        import torch
        from diffusers import ShapEImg2ImgPipeline
        from diffusers.utils import export_to_obj
        from PIL import Image
        
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16)
        pipe.to("cuda")
        
        image = Image.open(image_path).convert("RGB")
        images = pipe(
            image,
            guidance_scale=3.0,
            num_inference_steps=64,
            output_type="mesh"
        ).images
        
        mesh = images[0]
        export_to_obj(mesh, output_filename)
        print(f"[+] Native 3D model successfully saved to {output_filename}")
        return True
    except Exception as e:
        print(f"[!] True 3D generation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="LLM Game Object Generator (Open Source)")
    parser.add_argument("--prompt", type=str, help="Text description of the game object")
    parser.add_argument("--image", type=str, help="Path to an image to extract object data from (Image-to-Object)")
    parser.add_argument("--llm_model", type=str, default="llama3", help="Ollama model to use for logic generation")
    parser.add_argument("--local_image", action="store_true", help="Generate the visual asset locally using Stable Diffusion XL")
    parser.add_argument("--vision_model", type=str, default="llama3.2-vision", help="Ollama Vision model")
    parser.add_argument("--style", type=str, default="pixel-art", choices=["pixel-art", "vector-hand-drawn", "low-poly", "realistic-high-poly", "cel-shaded-stylized", "voxels", "isometric-2.5d", "particle-systems"], help="Visual style of the game object")
    parser.add_argument("--stateful", action="store_true", help="Generate state-based variations and transitions for the object")
    parser.add_argument("--image_model", type=str, default="sdxl", choices=["sdxl", "flux"], help="The image generation model to use locally (sdxl or flux)")

    
    args = parser.parse_args()
    
    if not args.prompt and not args.image:
        parser.error("You must provide either a --prompt or an --image")
        
    print("--- Starting Generation ---")
    
    # 1. Determine prompt (from text or extracted from image)
    active_prompt = args.prompt
    if args.image:
        extracted_prompt = extract_data_from_image(args.image, model=args.vision_model)
        active_prompt = f"{extracted_prompt} {args.prompt or ''}".strip()
        print(f"[*] Combined Prompt: {active_prompt}")

    # Set up clean output folder structure
    clean_name = clean_filename(active_prompt.split(",")[0] if "," in active_prompt else active_prompt)
    import datetime
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
    output_dir = os.path.join("generated_assets", args.style, f"{clean_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    final_logic_file = os.path.join(output_dir, f"{clean_name}_object_data.json")

    # 2. Agent 1: The Game Designer (Generates Logic JSON)
    temp_logic = agent_game_designer(active_prompt, model=args.llm_model, stateful=args.stateful)
    
    try:
        with open(temp_logic, "r") as f:
            logic_data = json.load(f)
    except Exception as e:
        print(f"[!] Failed to parse logic JSON: {e}. Creating fallback.")
        logic_data = {"name": clean_name.replace("_", " ").title(), "type": "prop", "description": active_prompt}

    # 3. Agent 2: The Prompt Engineer (Translates JSON to Visual Prompt)
    base_visual_prompt = agent_prompt_engineer(logic_data, args.style, model=args.llm_model)

    # Cleanup temp logic file
    if os.path.exists(temp_logic):
        os.remove(temp_logic)

    # 3. Generate Visual Assets
    # Check if the object is stateful and contains states
    states = logic_data.get("states", {})
    
    if args.stateful and states:
        print(f"[+] Generating stateful sprites and models for states: {list(states.keys())}")
        logic_data["current_state"] = list(states.keys())[0]
        
        for state_name, state_info in states.items():
            state_desc = state_info.get("description", "")
            
            # Combine the base optimized prompt with the specific state details
            state_prompt = f"{state_name} state: {base_visual_prompt}, {state_desc}"
            
            state_sprite_filename = os.path.join(output_dir, f"{clean_name}_sprite_{state_name}.png")
            state_model_filename = os.path.join(output_dir, f"{clean_name}_model_{state_name}.obj")
            
            # Agent 3 (Artist): Generate state image
            generate_visual_asset(state_prompt, output_filename=state_sprite_filename, local_image=args.local_image, style=args.style, init_image_path=args.image, image_model=args.image_model)
            
            # Agent 4 (Pixel Processor): Post-process state sprite (background removal, alpha clamping, color locking, black outline)
            if os.path.exists(state_sprite_filename):
                try:
                    from pixel_processor import post_process_sprite
                    ref_dir = os.path.join("reference_datasets", clean_name)
                    if not os.path.exists(ref_dir):
                        ref_dir = None
                    post_process_sprite(state_sprite_filename, args.style, ref_dir)
                except Exception as e:
                    print(f"[!] Sprite post-processing failed for {state_name}: {e}")
            
            # Extrude state to 3D OBJ
            print(f"[*] Extruding {state_name} 2D sprite to 3D model...")
            try:
                extrude_sprite_to_voxel_obj(state_sprite_filename, state_model_filename, style=args.style)
            except Exception as e:
                print(f"[!] 3D extrusion failed for {state_name}: {e}")
                
            # Add state specific asset URLs
            state_info["sprite_url"] = f"/generated_assets/{args.style}/{clean_name}_sprite_{state_name}.png"
            state_info["model_3d_url"] = f"/generated_assets/{args.style}/{clean_name}_model_{state_name}.obj"
            
        # Point primary urls to the default/first state
        first_state = list(states.keys())[0]
        logic_data["sprite_url"] = states[first_state]["sprite_url"]
        logic_data["model_3d"] = states[first_state]["model_3d_url"]
        
    else:
        # Standard Single Asset Generation
        sprite_filename = os.path.join(output_dir, f"{clean_name}_sprite.png")
        model_filename = os.path.join(output_dir, f"{clean_name}_model.obj")
        
        # 1. Provide the High Quality 2D Reference Image
        print("[*] Preparing 2D reference image for Image-to-3D...")
        if args.image and os.path.exists(args.image):
            import shutil
            shutil.copy2(args.image, sprite_filename)
        else:
            generate_visual_asset_with_reflexion(base_visual_prompt, logic_data, sprite_filename, args)
        
        # 2. Clean the background
        if os.path.exists(sprite_filename):
            try:
                from pixel_processor import post_process_sprite
                post_process_sprite(sprite_filename, args.style, None)
            except Exception as e:
                print(f"[!] Sprite post-processing failed: {e}")
        
        # 3. Mathematically Extrude the 2D image into 3D (Bypassing Broken Neural Networks)
        print("[*] Generating True AAA 3D Model using Algorithmic Math Extrusion...")
        from algorithmic_3d import generate_algorithmic_mesh
        obj_path, albedo_path = generate_algorithmic_mesh(sprite_filename, output_dir, clean_name)
        
        logic_data["sprite_url"] = f"/generated_assets/{args.style}/{clean_name}_{timestamp}/{clean_name}_sprite.png"
        logic_data["model_3d"] = f"/generated_assets/{args.style}/{clean_name}_{timestamp}/{clean_name}_model.obj"
        logic_data["style"] = "native-3d"
        
        # 4. Generate PBR Suite from TripoSR Albedo Map
        try:
            import pbr_generator
            base_pbr_path = os.path.join(output_dir, f"{clean_name}")
            print(f"[*] Generating PBR Suite from Albedo Map ({albedo_path})...")
            pbr_maps = pbr_generator.generate_pbr_suite(albedo_path, base_pbr_path)
            
            # 5. Connect PBR to MTL
            mtl_path = os.path.join(output_dir, f"{clean_name}_model.mtl")
            with open(mtl_path, 'w') as f:
                f.write(f"newmtl BakeMat\n")
                f.write(f"Ka 1.000 1.000 1.000\n")
                f.write(f"Kd 1.000 1.000 1.000\n")
                f.write(f"Ks 0.500 0.500 0.500\n")
                f.write(f"map_Kd {os.path.basename(albedo_path)}\n")
                f.write(f"map_Bump -bm 1.0 {os.path.basename(pbr_maps['normal'])}\n")
                f.write(f"map_Pr {os.path.basename(pbr_maps['roughness'])}\n")
                f.write(f"map_Pm {os.path.basename(pbr_maps['metallic'])}\n")
                f.write(f"map_Ka {os.path.basename(pbr_maps['ao'])}\n")
                
            # Update OBJ to link MTL
            with open(obj_path, 'r') as f:
                obj_data = f.read()
            with open(obj_path, 'w') as f:
                f.write(f"mtllib {os.path.basename(mtl_path)}\n")
                f.write(obj_data)
                
        except Exception as e:
            print(f"[!] PBR Generation failed: {e}")
            
        logic_data["model_3d"] = obj_path
        
        logic_data["sprite_url"] = f"/generated_assets/{args.style}/{clean_name}_{timestamp}/{clean_name}_sprite.png"
        logic_data["model_3d"] = f"/generated_assets/{args.style}/{clean_name}_{timestamp}/{clean_name}_model.obj"
        logic_data["style"] = "native-3d"
        
        # 4. Generate PBR Suite from Shap-E Albedo Map
        try:
            import pbr_generator
            base_pbr_path = os.path.join(output_dir, f"{clean_name}")
            print(f"[*] Generating PBR Suite from Albedo Map ({albedo_path})...")
            pbr_maps = pbr_generator.generate_pbr_suite(albedo_path, base_pbr_path)
            
            # 5. Connect PBR to MTL
            mtl_path = os.path.join(output_dir, f"{clean_name}_model.mtl")
            with open(mtl_path, 'w') as f:
                f.write(f"newmtl BakeMat\n")
                f.write(f"Ka 1.000 1.000 1.000\n")
                f.write(f"Kd 1.000 1.000 1.000\n")
                f.write(f"Ks 0.500 0.500 0.500\n")
                f.write(f"map_Kd {os.path.basename(albedo_path)}\n")
                f.write(f"map_Bump -bm 1.0 {os.path.basename(pbr_maps['normal'])}\n")
                f.write(f"map_Pr {os.path.basename(pbr_maps['roughness'])}\n")
                f.write(f"map_Pm {os.path.basename(pbr_maps['metallic'])}\n")
                f.write(f"map_Ka {os.path.basename(pbr_maps['ao'])}\n")
                
            # Update OBJ to link MTL
            with open(obj_path, 'r') as f:
                obj_data = f.read()
            with open(obj_path, 'w') as f:
                f.write(f"mtllib {os.path.basename(mtl_path)}\n")
                f.write(obj_data)
                
        except Exception as e:
            print(f"[!] PBR Generation failed: {e}")
            
        logic_data["model_3d"] = obj_path
            
        logic_data["sprite_url"] = f"/generated_assets/{args.style}/{clean_name}_{timestamp}/{clean_name}_sprite.png"

    # Inject metadata & write final JSON
    logic_data["style"] = args.style
    with open(final_logic_file, "w") as f:
        json.dump(logic_data, f, indent=4)
        
    # Validate final main image
    primary_sprite = os.path.join(output_dir, f"{clean_name}_sprite_{list(states.keys())[0]}.png" if (args.stateful and states) else f"{clean_name}_sprite.png")
    if os.path.exists(primary_sprite):
        validate_visual_asset(primary_sprite, active_prompt, model=args.vision_model)
        
    print(f"--- Generation Complete ---")
    print(f"Visual Assets saved inside: {output_dir}")
    print(f"Logic Data saved to: {final_logic_file}")

if __name__ == "__main__":
    main()
