import os
import json
import shutil

def main():
    base_dir = "generated_assets"
    if not os.path.exists(base_dir):
        print("[!] No generated_assets folder found. Nothing to reorganize.")
        return

    print("[*] Starting asset reorganization...")
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # If this is already reorganized (it contains style subfolders instead of direct asset files)
        # We can check by seeing if it contains any directory that matches our styles
        style_subfolders = ["pixel-art", "vector-hand-drawn", "low-poly", "realistic-high-poly", "cel-shaded-stylized", "voxels", "isometric-2.5d", "particle-systems"]
        has_subfolders = any(os.path.isdir(os.path.join(folder_path, s)) for s in style_subfolders)
        if has_subfolders:
            print(f"[i] Folder '{folder}' already contains style subfolders. Skipping.")
            continue
            
        # Find style of this asset
        json_path = os.path.join(folder_path, "object_data.json")
        style = "pixel-art" # default
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                style = data.get("style", "pixel-art")
            except Exception as e:
                print(f"[!] Error reading JSON for '{folder}': {e}")
                
        print(f"[*] Reorganizing '{folder}' -> style subfolder '{style}'")
        
        # Create style subdirectory
        style_dir = os.path.join(folder_path, style)
        os.makedirs(style_dir, exist_ok=True)
        
        # Move all files into style dir
        for filename in os.listdir(folder_path):
            file_src = os.path.join(folder_path, filename)
            if filename == style or os.path.isdir(file_src):
                continue
                
            file_dst = os.path.join(style_dir, filename)
            shutil.move(file_src, file_dst)
            
        # Update metadata inside the new style folder
        new_json_path = os.path.join(style_dir, "object_data.json")
        new_obj_path = os.path.join(style_dir, "model.obj")
        
        if os.path.exists(new_json_path):
            try:
                with open(new_json_path, "r") as f:
                    data = json.load(f)
                data["style"] = style
                if os.path.exists(new_obj_path):
                    data["model_3d"] = f"/generated_assets/{folder}/{style}/model.obj"
                with open(new_json_path, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"[+] Reorganized '{folder}' successfully.")
            except Exception as e:
                print(f"[!] Failed to update JSON for reorganized '{folder}': {e}")

if __name__ == "__main__":
    main()
