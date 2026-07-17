import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
import urllib.request
from PIL import Image

try:
    print("Loading Shap-E model...")
    pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16)
    pipe.to("cuda")
    print("Model loaded. Testing generation...")
    image = Image.new("RGB", (256, 256), (255, 0, 0))
    generator = torch.Generator(device="cuda").manual_seed(0)
    images = pipe(image, generator=generator, guidance_scale=3.0, num_inference_steps=64, output_type="mesh").images
    print("Exporting...")
    export_to_obj(images[0], "test_shape_output.obj")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
