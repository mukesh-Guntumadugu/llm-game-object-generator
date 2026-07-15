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

## Supported Art Styles

To ensure consistency in game engine integration, this pipeline will support multiple visual styles. However, to achieve maximum precision, **we are focusing on and mastering one style at a time**:

1. **2D Pixel Art (CURRENT FOCUS):** Retro 2D sprites with visible pixel grids, flat coloring, and bold outlines (similar to Stardew Valley, Celeste).
2. **2D Vector / Hand-drawn:** Clean vector curves, smooth cartoon shading, and digital illustrations (similar to Hollow Knight, Cuphead).
3. **3D Low-Poly:** Chunky, geometric models with a low polygon count; often uses flat colors (similar to Old School RuneScape, classic PS1 games).
4. **3D Realistic / High-Poly:** Highly detailed models designed to closely mimic real life and physics (similar to The Last of Us, Cyberpunk 2077).
5. **3D Cel-Shaded / Stylized:** 3D models shaded to look like a 2D comic book or cartoon (similar to Breath of the Wild, Borderlands).
6. **Voxels:** 3D cubes stacked together to create objects (similar to Minecraft, Crossy Road).
7. **2.5D / Isometric:** 2D sprites drawn and angled to give the illusion of 3D depth (similar to classic Fallout, Hades, Diablo).
8. **Particle Systems:** Collections of tiny, moving images or meshes used to create visual effects (similar to fire, smoke, rain, explosions).

> [!NOTE]
> **Current Pipeline Focus:** We are currently conquering and optimizing for **Style 1: 2D Pixel Art**. All prompt templates, SDXL checkpoints, and post-processing tools (including background removal) are tuned for clean 2D pixel-art assets.

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
