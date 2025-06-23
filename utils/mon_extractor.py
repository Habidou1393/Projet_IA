import os
import bz2
import re
import json

INPUT = "data/frwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = "data/frwiki_extrait"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_articles():
    with bz2.open(INPUT, 'rt', encoding='utf-8') as f:
        article = []
        in_article = False
        title = ""

        for line in f:
            if "<page>" in line:
                in_article = True
                article = []
            if in_article:
                article.append(line)
            if "</page>" in line:
                in_article = False
                full_article = "".join(article)
                title_match = re.search(r"<title>(.*?)</title>", full_article)
                text_match = re.search(r"<text[^>]*>(.*?)</text>", full_article, re.DOTALL)
                ns_match = re.search(r"<ns>(\d+)</ns>", full_article)

                if ns_match and ns_match.group(1) == "0" and title_match and text_match:
                    title = title_match.group(1)
                    text = text_match.group(1).strip()
                    save_article(title, text)

def save_article(title, text):
    safe_title = re.sub(r'[^\w\-_. ]', '_', title)[:100]
    filename = os.path.join(OUTPUT_DIR, f"{safe_title}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "title": title,
            "text": text,
            "ns": 0
        }, f, ensure_ascii=False)

if __name__ == "__main__":
    extract_articles()
