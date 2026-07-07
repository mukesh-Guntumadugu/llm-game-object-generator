import os
import json
import argparse
import base64
import requests
from PIL import Image
from io import BytesIO

# Configuration for Open Source Models
# You can use a local Ollama instance for the LLM or Hugging Face.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

def generate_visual_asset(prompt: str, output_filename: str = "sprite.png") -> str:
    """
    Uses an open-source diffusion model (e.g., Stable Diffusion via Hugging Face Inference API)
    to generate the game object sprite.
    """
    print(f"[*] Generating image using Open Source Diffusion Model for: '{prompt}'")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Refining the prompt for 2D game asset style
    refined_prompt = f"2d game asset, sprite, single object, white background, {prompt}"
    
    try:
        if not HF_API_TOKEN:
            print("[!] Warning: HF_API_TOKEN not set. Skipping real API call and creating a mock image.")
            img = Image.new('RGB', (512, 512), color = (73, 109, 137))
            img.save(output_filename)
            return output_filename
            
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": refined_prompt})
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image.save(output_filename)
        return output_filename
    except Exception as e:
        print(f"[!] Image generation failed: {e}")
        return output_filename

def extract_data_from_image(image_path: str) -> str:
    """
    Placeholder for extracting data/concept from an existing image (Image-to-Object).
    This could use an open-source vision-language model like LLaVA.
    """
    print(f"[*] Extracting object concept from image: {image_path}")
    # In a real scenario, you would send this image to LLaVA via Ollama or HF.
    return "Extracted object description based on image."

def generate_logic_and_behavior(prompt: str, model: str = "llama3") -> str:
    """
    Uses an open-source LLM (e.g., Llama 3 via Ollama) to generate the object's properties and logic.
    Always outputs engine-agnostic JSON.
    """
    print(f"[*] Generating logic using Open Source LLM ({model}) for: '{prompt}'")
    
    system_prompt = (
        "You are a game development AI. Generate the properties and behavior of a game object based on the user's description. "
        "Output ONLY valid JSON representing the object. Do not include markdown blocks or any other text. "
        "Include fields for: name, type, hp (if applicable), speed, hitboxes (width, height), and interactions (list of strings)."
    )
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nDescription: {prompt}",
        "stream": False,
        "format": "json"
    }
    
    output_filename = "object_data.json"
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        result_json = response.json().get("response", "{}")
        
        # Save to file
        with open(output_filename, "w") as f:
            f.write(result_json)
            
        return output_filename
    except requests.exceptions.ConnectionError:
        print("[!] Could not connect to local Ollama instance. Is it running? Creating mock JSON.")
        mock_data = {
            "name": "Mock Object",
            "type": "prop",
            "hp": 100,
            "speed": 0,
            "hitboxes": {"width": 32, "height": 32},
            "interactions": ["None"]
        }
        with open(output_filename, "w") as f:
            json.dump(mock_data, f, indent=4)
        return output_filename
    except Exception as e:
        print(f"[!] Logic generation failed: {e}")
        return output_filename

def main():
    parser = argparse.ArgumentParser(description="LLM Game Object Generator (Open Source)")
    parser.add_argument("--prompt", type=str, help="Text description of the game object")
    parser.add_argument("--image", type=str, help="Path to an image to extract object data from (Image-to-Object)")
    parser.add_argument("--llm_model", type=str, default="llama3", help="Ollama model to use for logic generation (e.g., llama3, mistral)")
    
    args = parser.parse_args()
    
    if not args.prompt and not args.image:
        parser.error("You must provide either a --prompt or an --image")
        
    print("--- Starting Generation ---")
    
    # 1. Determine prompt (from text or extracted from image)
    active_prompt = args.prompt
    if args.image:
        extracted_prompt = extract_data_from_image(args.image)
        active_prompt = f"{extracted_prompt} {args.prompt or ''}".strip()
        print(f"[*] Combined Prompt: {active_prompt}")

    # 2. Generate Assets
    image_file = generate_visual_asset(active_prompt)
    logic_file = generate_logic_and_behavior(active_prompt, model=args.llm_model)
    
    print(f"--- Generation Complete ---")
    print(f"Visual Asset saved to: {image_file}")
    print(f"Logic Data saved to: {logic_file}")
    print("Note: Output is engine-agnostic JSON. You can easily write adapters for Unity or Pygame.")

if __name__ == "__main__":
    main()
