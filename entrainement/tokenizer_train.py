import os
import sys
import argparse
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(
    input_files,
    vocab_size=5000,
    min_frequency=2,
    save_dir="tokenizer"
):
    # Vérifier la présence des fichiers d'entrée
    for f in input_files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Fichier d'entraînement introuvable : {f}")

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    if vocab_size < len(special_tokens):
        raise ValueError(f"Le vocabulaire doit contenir au moins les tokens spéciaux ({len(special_tokens)})")

    tokenizer = ByteLevelBPETokenizer()

    print(f"Démarrage de l'entraînement du tokenizer sur {input_files}...")
    tokenizer.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )

    # Création du dossier de sauvegarde si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde du tokenizer
    tokenizer.save_model(save_dir)
    print(f"Tokenizer entraîné et sauvegardé dans '{save_dir}/'")

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un tokenizer ByteLevel BPE")
    parser.add_argument("--input_files", nargs="+", default=["data/corpus_formate.txt"],
                        help="Liste des fichiers texte pour entraîner le tokenizer")
    parser.add_argument("--vocab_size", type=int, default=5000,
                        help="Taille du vocabulaire")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="Fréquence minimale d'un token pour être inclus")
    parser.add_argument("--save_dir", type=str, default="tokenizer",
                        help="Répertoire où sauvegarder le tokenizer entraîné")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        train_tokenizer(
            input_files=args.input_files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            save_dir=args.save_dir,
        )
    except Exception as e:
        print(f"Erreur lors de l'entraînement du tokenizer : {e}", file=sys.stderr)
        sys.exit(1)
