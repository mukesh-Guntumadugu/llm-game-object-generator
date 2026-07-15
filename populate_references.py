import csv
import requests
import time
import os

CSV_PATH = "dataset/master_game_objects.csv"

def search_wikimedia(query):
    """Searches Wikimedia Commons for a real photo of the queried noun."""
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 1,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    headers = {"User-Agent": "GameObjectAssetGenerator/1.0 (contact@gameassetgenerator.local)"}
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            imageinfo = page_info.get("imageinfo", [])
            if imageinfo:
                return imageinfo[0].get("url")
    except Exception as e:
        pass
    return None

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] File {CSV_PATH} not found.")
        return
        
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Cache URLs by the base noun to prevent spamming Wikipedia for duplicates
    cache = {}
    updated_count = 0
    
    print(f"[*] Scanning {len(rows)} rows for empty Reference_URLs...")
    
    for row in rows:
        if not row.get("Reference_URL"):
            # Extract base noun (the last word is typically the object, e.g., "Rusty Iron Sword" -> "Sword")
            prompt = row.get("Prompt", "")
            base_noun = prompt.split()[-1] if prompt else ""
            
            if base_noun:
                if base_noun not in cache:
                    url = search_wikimedia(base_noun)
                    cache[base_noun] = url if url else ""
                    time.sleep(0.1) # Be nice to the API
                
                url = cache[base_noun]
                if url:
                    row["Reference_URL"] = url
                    updated_count += 1
                    
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"[+] Successfully pulled and injected {updated_count} Wikipedia reference URLs into the dataset!")

if __name__ == "__main__":
    main()
