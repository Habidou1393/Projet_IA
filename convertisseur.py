import os
import json

input_dir = "data/frwiki_extrait"
output_file = "data/frwiki_articles.txt"

def is_main_namespace(page):
    return page.get('ns', -1) == 0

def main():
    with open(output_file, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        page = json.load(f)
                        if is_main_namespace(page):
                            text = page.get("text", "").strip()
                            if text:
                                out_f.write(text + "\n\n")
                    except json.JSONDecodeError:
                        continue

if __name__ == "__main__":
    main()
