import os
import torch
from pathlib import Path

from utils.wikipedia_search import recherche_wikipedia
from utils.google_search import recherche_google
from app.config import WIKI_TRIGGER, GOOGLE_TRIGGER

from entrainement.model import MiniGPT
from tokenizers import ByteLevelBPETokenizer

# Chargement tokenizer et modèle MiniGPT
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
VOCAB_SIZE = tokenizer.get_vocab_size()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(VOCAB_SIZE, block_size=64).to(device)

model_path = "entrainement/model-output/minigpt.pth"
if Path(model_path).exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modèle chargé depuis {model_path}")
else:
    print(f"Attention : modèle non trouvé à {model_path}")

model.eval()

try:
    eos_token_id = tokenizer.token_to_id("</s>")
except KeyError:
    eos_token_id = None

def generate_with_miniGPT(prompt: str, max_length=100) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([generated[-64:]], device=device)
            logits = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

    output = tokenizer.decode(generated)
    # Retourner uniquement la réponse générée (après le prompt)
    return output[len(prompt):].strip()


def obtenir_la_response(message: str) -> str:
    msg = message.strip()
    if not msg:
        return "Je n'ai pas compris ta question."

    # Recherche Wikipédia
    if msg.lower().startswith(WIKI_TRIGGER):
        query = msg[len(WIKI_TRIGGER):].strip()
        if not query:
            return "Tu dois préciser ce que tu veux chercher sur Wikipédia."
        try:
            res = recherche_wikipedia(query)
            return f"Résultat Wikipédia :\n{res}" if res else "Aucune information trouvée sur Wikipédia."
        except Exception as e:
            return f"Erreur Wikipédia : {e}"

    # Recherche Google
    if msg.lower().startswith(GOOGLE_TRIGGER):
        query = msg[len(GOOGLE_TRIGGER):].strip()
        if not query:
            return "Tu dois préciser ce que tu veux chercher sur Google."
        try:
            res = recherche_google(query)
            return f"Résultat Google :\n{res}" if res else "Aucune information trouvée via Google."
        except Exception as e:
            return f"Erreur Google : {e}"

    # Génération via ta propre IA MiniGPT
    prompt = f"Utilisateur : {msg}\nAssistant :"
    try:
        response = generate_with_miniGPT(prompt)
        return response.strip()
    except Exception as e:
        return f"Erreur génération IA : {e}"



if __name__ == "__main__":
    print("MiniGPT test interactif. Tape 'exit' ou 'quit' pour quitter.")
    while True:
        user_input = input("Utilisateur > ")
        if user_input.lower() in ("quit", "exit"):
            break
        print("Assistant >", generate_with_miniGPT(f"Utilisateur : {user_input}\nAssistant :"))
