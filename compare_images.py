import requests
from PIL import Image
from io import BytesIO

def main():
    print("[*] Loading generated image: sprite.png")
    try:
        gen_image = Image.open("sprite.png")
    except Exception as e:
        print(f"[!] Could not open sprite.png: {e}")
        return

    # URL to a clean public domain anvil image on Wikimedia Commons
    ref_url = "https://raw.githubusercontent.com/InventivetalentDev/minecraft-assets/1.19/assets/minecraft/textures/block/anvil.png"
    print(f"[*] Downloading reference image from internet: {ref_url}")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(ref_url, headers=headers)
        response.raise_for_status()
        ref_image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[!] Failed to download reference image: {e}")
        return

    # Standardize sizes to 512x512
    gen_image = gen_image.resize((512, 512))
    ref_image = ref_image.resize((512, 512))

    # Create a new blank image side-by-side (1024 width, 512 height)
    comparison = Image.new("RGBA", (1024, 512), color=(255, 255, 255, 255))

    # Paste generated image on the left, reference on the right
    comparison.paste(gen_image, (0, 0))
    comparison.paste(ref_image, (512, 0))

    # Save the output file
    output_filename = "comparison.png"
    comparison.save(output_filename)
    print(f"[+] Saved comparison image to: {output_filename}")

if __name__ == "__main__":
    main()
