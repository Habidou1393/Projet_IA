import re

input_path = "data/frwiki_articles_clean.txt"
output_path = "data/frwiki_articles_nettoye.txt"

def nettoyer_texte(texte):
    # Supprimer les redirections
    texte = re.sub(r'#REDIRECT.*', '', texte, flags=re.IGNORECASE)
    texte = re.sub(r'#REDIRECTION.*', '', texte, flags=re.IGNORECASE)

    # Supprimer les modèles {{...}}
    texte = re.sub(r'\{\{[^{}]*\}\}', '', texte)

    # Supprimer les balises HTML résiduelles
    texte = texte.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    # Supprimer les titres de sections (optionnel)
    texte = re.sub(r'==+ *.*? *==+', '', texte)

    # Supprimer les listes wiki inutiles
    texte = re.sub(r'^\*.*$', '', texte, flags=re.MULTILINE)
    texte = re.sub(r'^\:.*$', '', texte, flags=re.MULTILINE)

    # Supprimer les lignes vides en excès
    texte = re.sub(r'\n{2,}', '\n\n', texte)

    return texte.strip()

def main():
    with open(input_path, "r", encoding="utf-8") as in_f:
        texte = in_f.read()
    
    texte_nettoye = nettoyer_texte(texte)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(texte_nettoye)

    print(f"✅ Texte nettoyé sauvegardé dans : {output_path}")

if __name__ == "__main__":
    main()
