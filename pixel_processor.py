import os
import math
import numpy as np
from PIL import Image

def extract_dataset_palette(ref_dir=None, num_colors=16, img_path=None):
    """
    Extracts a list of RGB color tuples from reference images using K-Means clustering.
    If no reference images exist, it dynamically extracts the palette directly from the generated sprite.
    """
    all_pixels = []
    
    # 1. Try to gather pixels from reference directory
    if ref_dir and os.path.exists(ref_dir) and os.path.isdir(ref_dir):
        files = [
            os.path.join(ref_dir, f) 
            for f in os.listdir(ref_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        for f in files:
            try:
                with Image.open(f) as img:
                    small = img.convert("RGB").resize((32, 32))
                    arr = np.array(small)
                    inner = arr[4:28, 4:28]
                    flat = inner.reshape(-1, 3)
                    for r, g, b in flat:
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        if 15 < lum < 240:
                            all_pixels.append((r, g, b))
            except Exception:
                pass

    # 2. If no references, extract dynamically from the sprite itself
    if len(all_pixels) < 50 and img_path and os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGBA")
                arr = np.array(img)
                # Only extract colors from opaque pixels
                mask = arr[:, :, 3] > 100
                rgb = arr[:, :, :3][mask]
                
                # Sample pixels to keep performance fast
                if len(rgb) > 5000:
                    indices = np.random.choice(len(rgb), 5000, replace=False)
                    rgb = rgb[indices]
                    
                for r, g, b in rgb:
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    if 15 < lum < 240:
                        all_pixels.append((r, g, b))
        except Exception:
            pass

    # 3. Final safety net (just basic grayscale if completely transparent)
    if len(all_pixels) < 10:
         return [(0, 0, 0), (255, 255, 255), (128, 128, 128)]
         
    try:
        from scipy.cluster.vq import kmeans
        pixel_data = np.array(all_pixels, dtype=float)
        
        centroids, _ = kmeans(pixel_data, num_colors)
        palette = [tuple(map(int, np.round(c))) for c in centroids]
        
        # Ensure white is included as highlight
        if (255, 255, 255) not in palette:
            palette.append((255, 255, 255))
        return palette
    except Exception as e:
        print(f"[!] K-Means palette extraction failed: {e}")
        return [(0, 0, 0), (255, 255, 255), (128, 128, 128)]

def quantize_to_palette(img, palette_colors):
    """
    Maps all non-transparent pixels (alpha > 100) to their closest color in palette_colors.
    """
    if not palette_colors:
        return img
    img = img.convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    
    # Identify non-transparent region
    mask = alpha > 100
    if not np.any(mask):
        return img
        
    pixels = rgb[mask] # Shape: (N, 3)
    palette_arr = np.array(palette_colors) # Shape: (M, 3)
    
    # Broadcast subtraction to calculate Euclidean distances
    diff = pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2) # Shape: (N, M)
    closest_idx = np.argmin(dist_sq, axis=1) # Shape: (N,)
    
    # Map colors and enforce full opacity
    rgb[mask] = palette_arr[closest_idx]
    arr[:, :, :3] = rgb
    arr[mask, 3] = 255
    
    return Image.fromarray(arr)

def apply_black_outline(img):
    """
    Draws a clean 1-pixel solid black outline around the boundary of the alpha mask.
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3] > 100
    
    h, w = alpha.shape
    dilated = np.zeros_like(alpha)
    
    # Check orthogonal and diagonal neighbors (8-connectivity) for outline boundary
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(alpha)
            # Slices to avoid wrap-around
            sy_start = max(0, -dy)
            sy_end = min(h, h - dy)
            sx_start = max(0, -dx)
            sx_end = min(w, w - dx)
            
            ty_start = max(0, dy)
            ty_end = min(h, h + dy)
            tx_start = max(0, dx)
            tx_end = min(w, w + dx)
            
            shifted[ty_start:ty_end, tx_start:tx_end] = alpha[sy_start:sy_end, sx_start:sx_end]
            dilated = dilated | shifted
            
    outline = dilated & ~alpha
    
    # Draw solid black outline
    arr[outline] = [0, 0, 0, 255]
    
    return Image.fromarray(arr)

def post_process_sprite(sprite_path, style, ref_dir=None):
    """
    Applies premium post-processing filters (background removal, alpha clamping,
    color palette quantization, and clean black outline) to the sprite.
    """
    if not os.path.exists(sprite_path):
        return
        
    try:
        from rembg import remove, new_session
        bria_session = new_session("u2net")
    except ImportError:
        remove = None
        bria_session = None
        
    img = Image.open(sprite_path)
    
    # 1. Background removal
    if remove:
        try:
            img = remove(img, session=bria_session)
        except Exception as e:
            print(f"[!] rembg failed during post-processing: {e}")
            
    img = img.convert("RGBA")
    
    # 2. Alpha Clamping & Floating Noise Removal
    arr = np.array(img)
    a = arr[:, :, 3]
    a_new = np.where(a > 100, 255, 0).astype(np.uint8)
    
    # Use mathematical connected components to delete floating background chunks
    try:
        from scipy.ndimage import label, binary_opening
        
        # Sever microscopic 1-pixel bridges connecting distinct objects
        eroded_mask = binary_opening(a_new > 0, iterations=2)
        
        labeled_array, num_features = label(eroded_mask)
        if num_features > 1:
            # Find the largest component from the eroded mask
            sizes = np.bincount(labeled_array.ravel())
            sizes[0] = 0 # Ignore background
            largest_label = sizes.argmax()
            
            # Create a clean mask from the largest blob
            core_mask = (labeled_array == largest_label)
            
            # We want to keep the ORIGINAL pixels that fall under this core mask, 
            # plus we dilate it slightly back to its original size to not lose edges.
            from scipy.ndimage import binary_dilation
            restored_mask = binary_dilation(core_mask, iterations=2)
            
            a_new = np.where(restored_mask & (a_new > 0), 255, 0).astype(np.uint8)
    except Exception as e:
        print(f"[!] Floating noise removal failed: {e}")
        
    arr[:, :, 3] = a_new
    img = Image.fromarray(arr)
    
    # 3. Only apply palette quantization and black outlines to pixel-art/voxels styles
    if style in ["pixel-art", "voxels"]:
        # Extract palette from reference directory, or dynamically from the sprite itself
        palette = extract_dataset_palette(ref_dir, img_path=sprite_path)
        # Quantize colors
        img = quantize_to_palette(img, palette)
        # Apply clean black outline
        img = apply_black_outline(img)
        
    img.save(sprite_path)
    print(f"[+] Post-processed sprite successfully saved: {sprite_path}")

