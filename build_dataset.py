import csv
import os
import random

os.makedirs("dataset", exist_ok=True)

categories = {
    "Consumables": {
        "adjectives": ["Glowing", "Rotten", "Fresh", "Stale", "Enchanted", "Poisonous", "Spicy", "Sweet", "Mysterious", "Fizzy", "Bubbling", "Vile"],
        "materials": ["Red", "Blue", "Green", "Golden", "Silver", "Crystal", "Glass", "Wooden", "Iron", "Mud", "Ceramic"],
        "nouns": ["Potion", "Apple", "Steak", "Bread", "Cheese", "Stew", "Elixir", "Flask", "Mushroom", "Berry", "Drumstick"],
        "stateful": True,
        "count": 65
    },
    "Weapons": {
        "adjectives": ["Rusty", "Shiny", "Broken", "Ancient", "Legendary", "Cursed", "Mastercrafted", "Serrated", "Heavy", "Lightweight", "Bloodied", "Dull"],
        "materials": ["Iron", "Steel", "Wooden", "Diamond", "Obsidian", "Bronze", "Gold", "Laser", "Plasma", "Bone", "Silver", "Mithril"],
        "nouns": ["Broadsword", "Dagger", "Longbow", "Shotgun", "Rifle", "Staff", "Axe", "Spear", "Crossbow", "Wand", "Mace", "Katana"],
        "stateful": True,
        "count": 125
    },
    "Vehicles": {
        "adjectives": ["Rusted", "Sleek", "Armored", "Damaged", "Futuristic", "Vintage", "Hovering", "Flying", "Submersible", "Speedy", "Bulky", "Scorched"],
        "materials": ["Scrap Metal", "Carbon Fiber", "Steel", "Wooden", "Aluminum", "Neon", "Chrome", "Iron", "Plastic", "Glass"],
        "nouns": ["Pickup Truck", "Hoverbike", "Spaceship", "Rowboat", "Motorcycle", "Tank", "Helicopter", "Submarine", "Glider", "Kart", "Sedan"],
        "stateful": True,
        "count": 65
    },
    "Environment Props": {
        "adjectives": ["Mossy", "Cracked", "Pristine", "Burning", "Frozen", "Overgrown", "Shattered", "Dusty", "Glowing", "Abandoned", "Ancient", "Toppled"],
        "materials": ["Stone", "Wooden", "Brick", "Metal", "Marble", "Ice", "Mud", "Sandstone", "Obsidian", "Concrete", "Granite"],
        "nouns": ["Pillar", "Barrel", "Crate", "Statue", "Campfire", "Street Lamp", "Fountain", "Signpost", "Tombstone", "Fence", "Anvil"],
        "stateful": True,
        "count": 65
    },
    "Furniture": {
        "adjectives": ["Antique", "Modern", "Broken", "Sturdy", "Plush", "Dusty", "Elegant", "Cheap", "Cozy", "Gothic", "Torn", "Stained"],
        "materials": ["Oak", "Leather", "Mahogany", "Pine", "Plastic", "Metal", "Glass", "Stone", "Velvet", "Wicker"],
        "nouns": ["Armchair", "Bookshelf", "Bed", "Table", "Desk", "Throne", "Cabinet", "Couch", "Stool", "Wardrobe", "Chest"],
        "stateful": True,
        "count": 65
    },
    "Loot": {
        "adjectives": ["Sparkling", "Ancient", "Cursed", "Flawless", "Tarnished", "Mystical", "Heavy", "Fake", "Glimmering", "Hidden", "Polished"],
        "materials": ["Gold", "Silver", "Diamond", "Emerald", "Ruby", "Sapphire", "Crystal", "Pearl", "Amethyst", "Bronze", "Platinum"],
        "nouns": ["Coin", "Ring", "Necklace", "Goblet", "Crown", "Scroll", "Gemstone", "Idol", "Relic", "Chalice", "Medallion"],
        "stateful": False,
        "count": 65
    },
    "Tools": {
        "adjectives": ["Rusty", "Reliable", "Heavy", "Dull", "Sharp", "Broken", "Advanced", "Mechanical", "Magical", "Sturdy", "Bent", "Oiled"],
        "materials": ["Iron", "Steel", "Wooden", "Copper", "Titanium", "Bronze", "Flint", "Stone", "Diamond", "Laser"],
        "nouns": ["Pickaxe", "Shovel", "Fishing Rod", "Hammer", "Wrench", "Saw", "Blowtorch", "Crowbar", "Sickle", "Chisel", "Mallet"],
        "stateful": True,
        "count": 65
    },
    "Produce": {
        "adjectives": ["Ripe", "Rotten", "Giant", "Tiny", "Glowing", "Juicy", "Dried", "Bruised", "Perfect", "Poisonous", "Fresh", "Wilted"],
        "materials": ["Red", "Green", "Yellow", "Purple", "Orange", "Brown", "Blue", "Golden", "Rainbow", "Dark"],
        "nouns": ["Apple", "Banana", "Pumpkin", "Watermelon", "Carrot", "Potato", "Tomato", "Onion", "Grapes", "Corn", "Cabbage"],
        "stateful": True,
        "count": 65
    }
}

random.seed(42) # For reproducibility
unique_objects = set()
dataset = []

for category, data in categories.items():
    needed = data["count"]
    attempts = 0
    generated = 0
    while generated < needed and attempts < 2000:
        adj = random.choice(data["adjectives"])
        mat = random.choice(data["materials"])
        noun = random.choice(data["nouns"])
        
        if random.random() > 0.4:
            name = f"{adj} {mat} {noun}"
        else:
            name = f"{adj} {noun}"
            
        if name not in unique_objects:
            unique_objects.add(name)
            # Decide if stateful
            stateful = "true" if data["stateful"] and random.random() > 0.5 else "false"
            dataset.append({
                "Category": category,
                "Prompt": name,
                "Stateful": stateful,
                "Reference_URL": ""
            })
            generated += 1
        attempts += 1

# Save to CSV
csv_path = "dataset/master_game_objects.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Category", "Prompt", "Stateful", "Reference_URL"])
    writer.writeheader()
    writer.writerows(dataset)

print(f"Successfully generated {len(dataset)} unique game objects at {csv_path}")
