import cv2
import numpy as np
import os

def generate_normal_map(image_path, output_path, intensity=2.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    # Calculate X and Y gradients using Sobel filter
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize gradients
    normal_x = (sobel_x / 255.0) * intensity
    normal_y = (sobel_y / 255.0) * intensity
    normal_z = np.ones_like(normal_x)
    
    # Combine into vectors and normalize length
    length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= length
    normal_y /= length
    normal_z /= length
    
    # Map from [-1, 1] vector space to [0, 255] RGB color space
    r = ((normal_x + 1.0) * 127.5).astype(np.uint8)
    g = ((normal_y + 1.0) * 127.5).astype(np.uint8)
    b = ((normal_z + 1.0) * 127.5).astype(np.uint8)
    
    # Normal map is RGB (OpenCV uses BGR)
    normal_map = cv2.merge([b, g, r])
    cv2.imwrite(output_path, normal_map)

def generate_roughness_map(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    # Invert image (bright areas = specular highlights = smooth = dark roughness)
    roughness = 255 - img
    # Adjust contrast to make it more pronounced
    roughness = cv2.normalize(roughness, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(output_path, roughness)

def generate_metallic_map(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    # Very bright areas tend to be metallic reflections
    _, metallic = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    metallic = cv2.GaussianBlur(metallic, (5, 5), 0)
    cv2.imwrite(output_path, metallic)

def generate_ao_map(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    inverted = 255 - img
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Simulate ambient occlusion by multiplying inverted blurred edges
    ao = 255 - cv2.multiply(inverted, blurred, scale=1/255.0)
    ao = cv2.normalize(ao, None, 100, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, ao)

def generate_pbr_suite(sprite_path, base_output_path):
    """
    Generates a full PBR suite from a base color sprite.
    Returns a dictionary of paths to the generated maps.
    """
    print("[*] Generatic PBR Textures (Normal, Roughness, Metallic, AO)...")
    generate_normal_map(sprite_path, f"{base_output_path}_normal.png")
    generate_roughness_map(sprite_path, f"{base_output_path}_roughness.png")
    generate_metallic_map(sprite_path, f"{base_output_path}_metallic.png")
    generate_ao_map(sprite_path, f"{base_output_path}_ao.png")
    
    return {
        "albedo": os.path.basename(sprite_path),
        "normal": os.path.basename(f"{base_output_path}_normal.png"),
        "roughness": os.path.basename(f"{base_output_path}_roughness.png"),
        "metallic": os.path.basename(f"{base_output_path}_metallic.png"),
        "ao": os.path.basename(f"{base_output_path}_ao.png")
    }

def create_mtl_file(mtl_path, material_name, pbr_maps):
    """
    Writes a .mtl file mapping the PBR textures so game engines/3D viewers pick them up automatically.
    """
    mtl_content = f"newmtl {material_name}\n"
    mtl_content += "Ka 1.000000 1.000000 1.000000\n"
    mtl_content += "Kd 1.000000 1.000000 1.000000\n"
    mtl_content += "Ks 0.500000 0.500000 0.500000\n"
    mtl_content += "Ns 250.000000\n"
    mtl_content += "illum 2\n"
    
    # Texture maps
    if "albedo" in pbr_maps:
        mtl_content += f"map_Kd {pbr_maps['albedo']}\n"
    if "normal" in pbr_maps:
        mtl_content += f"map_Bump -bm 1.0 {pbr_maps['normal']}\n"
        mtl_content += f"norm {pbr_maps['normal']}\n"
    if "roughness" in pbr_maps:
        mtl_content += f"map_Pr {pbr_maps['roughness']}\n"
        mtl_content += f"map_Ns {pbr_maps['roughness']}\n"
    if "metallic" in pbr_maps:
        mtl_content += f"map_Pm {pbr_maps['metallic']}\n"
    if "ao" in pbr_maps:
        mtl_content += f"map_Ka {pbr_maps['ao']}\n"
        
    with open(mtl_path, "w") as f:
        f.write(mtl_content)
    print(f"[+] Material file created: {mtl_path}")
