import os
import json
import shutil
from PIL import Image
from generator import generate_visual_asset, clean_filename
from voxel_extruder import extrude_sprite_to_voxel_obj

try:
    from rembg import remove
except ImportError:
    remove = None

variations = [
    {"suffix": "golden", "desc": "magical golden apple, shining bright yellow gold, glowing aura", "type": "MagicFood", "hp": 50, "interactions": ["Eat", "Bless"]},
    {"suffix": "poisonous", "desc": "rotten poisonous apple, dripping green acid, bubbling purple surface", "type": "HazardFood", "hp": -25, "interactions": ["Eat", "Throw"]},
    {"suffix": "frozen", "desc": "frozen ice apple, crystalline translucent light blue ice, frost particles", "type": "Collectible", "hp": 5, "interactions": ["Eat", "Melt"]},
    {"suffix": "magma", "desc": "magma lava apple, burning dark obsidian rock with glowing red veins of lava", "type": "HazardProp", "hp": -10, "interactions": ["Touch", "Cool"]},
    {"suffix": "steampunk", "desc": "steampunk clockwork mechanical apple, polished brass gears, copper pipes, small rivets", "type": "Artifact", "hp": 0, "interactions": ["Examine", "Wind Up"]},
    {"suffix": "ghostly", "desc": "ghostly spectral apple, semi-transparent glowing green ectoplasm, ethereal smoke", "type": "MagicItem", "hp": 15, "interactions": ["Eat", "Exorcise"]},
    {"suffix": "cosmic", "desc": "cosmic galaxy apple, dark blue skin with glittering stars and purple nebulas", "type": "MagicFood", "hp": 100, "interactions": ["Eat", "Meditate"]},
    {"suffix": "cyberpunk", "desc": "cyberpunk digital apple, glowing neon pink lines, circuitry board pattern, holographic tint", "type": "TechProp", "hp": 2, "interactions": ["Hack", "Use"]},
    {"suffix": "diamond", "desc": "diamond gem apple, faceted crystal ruby red gem, sparkling reflections", "type": "Valuable", "hp": 0, "interactions": ["Sell", "Appraise"]},
    {"suffix": "wooden", "desc": "carved wooden apple toy, dark mahogany wood grain texture, polished finish", "type": "Toy", "hp": 0, "interactions": ["Play", "Burn"]},
    {"suffix": "void", "desc": "void shadow apple, swirling dark purple energy, black smoke, light-absorbing surface", "type": "HazardItem", "hp": -50, "interactions": ["Eat", "Banish"]},
    {"suffix": "slime", "desc": "slime gooey apple, green translucent gelatinous bubble, sticky drops", "type": "Edible", "hp": 5, "interactions": ["Eat", "Squeeze"]},
    {"suffix": "candy", "desc": "candy glazed apple, glossy red sugar coating, sprinkles on top, wooden stick", "type": "Food", "hp": 15, "interactions": ["Eat"]},
    {"suffix": "rotten", "desc": "decaying rotten apple, brown wrinkled skin, mold spots, small worm crawling out", "type": "Trash", "hp": -5, "interactions": ["Eat", "Compost"]},
    {"suffix": "crystal", "desc": "crystal quartz apple, transparent glass-like structure, rainbow prism refractions", "type": "Valuable", "hp": 0, "interactions": ["Examine", "Polish"]},
    {"suffix": "cacao", "desc": "cacao chocolate apple, dark brown chocolate coating, white chocolate drizzle", "type": "Food", "hp": 20, "interactions": ["Eat"]},
    {"suffix": "bomb", "desc": "ticking bomb apple, black metal fuse sticking out, sparks, danger warning label", "type": "Weapon", "hp": -100, "interactions": ["Ignite", "Throw"]},
    {"suffix": "leafy", "desc": "leafy forest apple, made of woven green leaves and fresh vines, dew drops", "type": "Collectible", "hp": 8, "interactions": ["Eat", "Plant"]},
    {"suffix": "stone", "desc": "rough stone apple, carved grey granite rock, moss growing on the sides", "type": "Prop", "hp": 0, "interactions": ["Examine", "Throw"]},
    {"suffix": "honey", "desc": "honey dipped apple, dripping golden honey, small honeybee flying nearby", "type": "Food", "hp": 30, "interactions": ["Eat"]},
    {"suffix": "undead", "desc": "undead zombie apple, pale green decaying skin, exposed bone-like seeds", "type": "HazardFood", "hp": -15, "interactions": ["Eat", "Purify"]},
    {"suffix": "bubblegum", "desc": "bubblegum pink apple, glossy plastic texture, colorful candy wrapper", "type": "Food", "hp": 10, "interactions": ["Eat", "Blow Bubble"]},
    {"suffix": "radioactive", "desc": "radioactive uranium apple, glowing lime green radiation symbols, bright green ooze", "type": "HazardProp", "hp": -80, "interactions": ["Inspect", "Store"]},
    {"suffix": "rainbow", "desc": "rainbow gradient apple, smooth transition from red to purple, glittering dust", "type": "MagicFood", "hp": 40, "interactions": ["Eat", "Wish"]},
    {"suffix": "iron", "desc": "rusted iron apple, cast iron metal plating, rust spots, heavy appearance", "type": "Prop", "hp": 0, "interactions": ["Examine", "Melt"]},
    {"suffix": "paper", "desc": "origami paper apple, folded red and green craft paper, sharp geometric folds", "type": "Art", "hp": 0, "interactions": ["Examine", "Unfold"]},
    {"suffix": "shadow", "desc": "shadow assassin apple, dark grey obsidian skin, smoke trail, quiet whispering sound", "type": "Artifact", "hp": 0, "interactions": ["Equip", "Inspect"]},
    {"suffix": "electric", "desc": "electric shock apple, glowing yellow skin with crackling blue lightning bolts", "type": "HazardItem", "hp": -30, "interactions": ["Eat", "Discharge"]},
    {"suffix": "cookie", "desc": "cookie apple, baked brown gingerbread dough, red icing details, chocolate chips", "type": "Food", "hp": 12, "interactions": ["Eat"]},
    {"suffix": "royal", "desc": "royal crowned apple, deep crimson skin, miniature gold crown on top, red velvet ribbon", "type": "Valuable", "hp": 25, "interactions": ["Wear", "Gift"]}
]

def generate_all_variations(limit=30):
    print(f"[*] Beginning Generation of {limit} Apple Variations...")
    
    style = "pixel-art"
    
    for i, var in enumerate(variations[:limit], start=1):
        suffix = var["suffix"]
        desc = var["desc"]
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        output_dir = os.path.join("generated_assets", style, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        sprite_path = os.path.join(output_dir, f"apple_{suffix}_sprite.png")
        model_path = os.path.join(output_dir, f"apple_{suffix}_model.obj")
        json_path = os.path.join(output_dir, f"apple_{suffix}_object_data.json")
        
        print(f"\n[{i}/{limit}] Generating Apple Variation: {suffix.upper()}")
        print(f"[*] Prompt description: {desc}")
        
        # Generate the sprite using generator logic
        generate_visual_asset(desc, output_filename=sprite_path, local_image=False, style=style)
        
        # Post-process sprite (background removal, alpha clamping, color locking, black outline)
        if os.path.exists(sprite_path):
            try:
                from pixel_processor import post_process_sprite
                post_process_sprite(sprite_path, style, ref_dir="reference_datasets/apple")
            except Exception as e:
                print(f"[!] Sprite post-processing failed for {suffix}: {e}")
                
        # Extrude to 3D OBJ
        print(f"[*] Extruding {suffix} 2D sprite to 3D model...")
        try:
            extrude_sprite_to_voxel_obj(sprite_path, model_path, style=style)
            print(f"[+] 3D model saved to: {model_path}")
        except Exception as e:
            print(f"[!] 3D extrusion failed for {suffix}: {e}")
            
        # Write metadata JSON
        meta = {
            "name": f"{suffix.title()} Apple",
            "type": var["type"],
            "hp": var["hp"],
            "speed": 0,
            "hitboxes": {"width": 1.0, "height": 1.0},
            "interactions": var["interactions"],
            "style": style,
            "sprite_url": f"/generated_assets/{style}/{timestamp}/apple_{suffix}_sprite.png",
            "model_3d": f"/generated_assets/{style}/{timestamp}/apple_{suffix}_model.obj"
        }
        
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[+] Metadata saved to: {json_path}")
        
    print("\n[+] Creative variations batch generation completed successfully!")

if __name__ == "__main__":
    generate_all_variations(30)
