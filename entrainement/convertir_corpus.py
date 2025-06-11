def convertir_corpus(input_path="data/mon_corpus.txt", output_path="data/corpus_formate.txt"):
    with open(input_path, "r", encoding="utf-8") as f:
        lignes = [ligne.strip() for ligne in f if ligne.strip()]  # Nettoyer lignes vides

    dialogues = []
    i = 0
    while i < len(lignes) - 1:
        question = lignes[i]
        reponse = lignes[i+1]

        # Ajouter préfixes
        dialogues.append(f"Utilisateur : {question}")
        dialogues.append(f"Assistant : {reponse}")
        dialogues.append("")  # Ligne vide pour séparer les dialogues

        i += 2  # On avance de 2 lignes (Q/R)

    # Écrire dans un nouveau fichier
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(dialogues))

    print(f"Corpus formaté sauvegardé dans {output_path}")


if __name__ == "__main__":
    convertir_corpus()
