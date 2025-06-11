import os
import torch
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.wikipedia_search import recherche_wikipedia
from utils.google_search import recherche_google
from app.config import WIKI_TRIGGER, GOOGLE_TRIGGER

# === Chargement du modèle fine-tuné GPT-2 ===
try:
    model_path = Path(__file__).resolve().parent.parent / "model-output"
    if not model_path.exists():
        raise FileNotFoundError(f"Le dossier du modèle n'existe pas : {model_path}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path.as_posix(), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_path.as_posix(), local_files_only=True)
    model.eval()

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle GPT-2 fine-tuné :\n{e}")

# === Génération de réponse GPT-2 ===
def generate_gpt2_response(prompt: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# === Fonction principale appelée par Flask ===
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

    # Génération via GPT-2
    prompt = f"Utilisateur : {msg}\nAssistant :"
    gpt2_output = generate_gpt2_response(prompt)
    return gpt2_output
