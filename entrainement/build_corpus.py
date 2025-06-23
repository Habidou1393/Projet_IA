import requests

def fetch_url_text(url):
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"Erreur {resp.status_code} sur {url}")
            return ""
    except Exception as e:
        print(f"Exception lors du fetch {url} : {e}")
        return ""

def build_corpus():
    sources = [
        # Classiques de la littérature française (extraits)
        "https://www.gutenberg.org/cache/epub/17489/pg17489.txt",  # Les Misérables - Victor Hugo
        "https://www.gutenberg.org/cache/epub/24234/pg24234.txt",  # Le Comte de Monte-Cristo - Alexandre Dumas

        # Textes variés public domain (extraits)
        "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",  # Les Fables de La Fontaine

        # Textes informatifs/didactiques
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # en anglais, juste pour exemple

        # Tu peux ajouter d'autres URL textuelles libres de droits ici
    ]

    corpus = ""
    for url in sources:
        print(f"Téléchargement de {url} ...")
        corpus += fetch_url_text(url) + "\n\n"

    print(f"Longueur totale du corpus : {len(corpus)} caractères")
    with open("data/corpus_formate.txt", "w", encoding="utf-8") as f:
        f.write(corpus)
    print("Corpus sauvegardé dans 'data/corpus_formate.txt'")

if __name__ == "__main__":
    build_corpus()
