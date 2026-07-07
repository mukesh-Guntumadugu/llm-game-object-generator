# LLM Game Object Generator

Welcome to the LLM Game Object Generator project!

This tool leverages **Open Source** Generative AI and Large Language Models (LLMs) to fully generate functional 2D game objects from text descriptions or existing images. Instead of just creating an image or a script, this generator outputs a complete package that includes both the visual representation (sprite) and the underlying data logic (behavior, interactions, boundaries). 

The generated output is **Engine-Agnostic** (JSON and PNG), meaning you can easily load these objects into any engine, including Unity and Pygame.

## Architecture & Concept

Generating functional game objects requires a two-step generation pipeline:

1. **Visual Asset Generation (Open Source Diffusion Models)**
   We use image generation APIs like **Stable Diffusion** (e.g., via Hugging Face Inference API or a local ComfyUI/Automatic1111 setup) to create 2D sprites based on the prompt. This stage focuses on aesthetics, art style, and background removal.

2. **Logic and Structure (Open Source Large Language Models)**
   We use local open-source LLMs (like **Llama 3** or **Mistral** via [Ollama](https://ollama.com/)) to generate structured engine-agnostic data (JSON). This stage defines:
   - Object properties (HP, type, speed)
   - Collision boundaries (hitboxes)
   - Behavior logic (how it walks, attacks, or interacts)

3. **Image-to-Object (Data Extraction)**
   Instead of relying solely on text prompts, you can provide a reference image. The orchestrator will extract the style and concept from the image (using Vision-Language models like LLaVA) and construct the game object based on it.

## Initial Setup

1. **Clone the repository** (if applicable).
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Setup Local LLM (Ollama)**:
   - Install [Ollama](https://ollama.com/)
   - Pull your preferred model: `ollama run llama3`
4. **Environment Variables**:
   Set up your Hugging Face API key in your environment if you want to use the serverless Stable Diffusion API:
   ```bash
   export HF_API_TOKEN="your_huggingface_token"
   ```

## Usage

**Text-to-Object**
```bash
python generator.py --prompt "A friendly NPC merchant selling potions" --llm_model llama3
```

**Image-to-Object**
```bash
python generator.py --image "concept_art.jpg" --prompt "make it a destructible prop"
```

Both commands will output a folder with `sprite.png` and `object_data.json`.

## Supported Engines
The generator natively outputs engine-agnostic `.json` files. You can write simple wrapper scripts in **Unity (C#)** or **Pygame (Python)** to parse this JSON and instantiate the visual assets in your game world.
