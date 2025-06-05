import os
import random
import torch
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.wikipedia_search import recherche_wikipedia
from utils.google_search import recherche_google
from app.config import WIKI_TRIGGER, GOOGLE_TRIGGER, MATH_TRIGGER

# === Chargement du modèle fine-tuné GPT-2 ===
try:
    model_path = Path(__file__).resolve().parent.parent / "model-output"
    if not model_path.exists():
        raise FileNotFoundError(f"Le dossier du modèle n'existe pas : {model_path}")
    
    print(f"[INFO] Chargement du modèle depuis : {model_path}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path.as_posix(), local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_path.as_posix(), local_files_only=True)
    model.eval()

except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle GPT-2 fine-tuné :\n{e}")

# === Réaction aléatoire (effet conversationnel) ===
def chatbot_reponse(texte: str) -> str:
    reactions = [
        "😊", "👍", "Ça me fait plaisir de t'aider !", "Super question !",
        "Tu es brillant(e) !", "Hmm...", "Intéressant...", "Voyons voir...",
        "C'est une bonne question.", "Je réfléchis...",
        "Je ne suis pas une boule de cristal, mais je crois que c'est ça ! 😂",
        "Si j'avais un euro à chaque fois qu'on me pose cette question... 💸",
        "Je suis un bot, mais je commence à comprendre les humains ! 🤖",
        "Je ne suis pas parfait, mais j'essaie ! 😅"
    ]
    return f"{random.choice(reactions)} {texte}"

# === Vérification si le message contient une salutation ===
def detection_salutation(message: str) -> str | None:
    greetings = ["bonjour", "salut", "coucou", "hello", "hey"]
    msg = message.lower().strip()

    if any(greeting in msg for greeting in greetings):
        return "Bonjour ! 😊"  # Réponse de salutation
    return None

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
        return "Je n'ai pas bien saisi ta question, pourrais-tu reformuler s'il te plaît ?"

    # Vérifier si c'est une salutation
    if (salutation := detection_salutation(msg)):
        return chatbot_reponse(salutation)

    # 🔍 Commandes spéciales
    if msg.lower().startswith(WIKI_TRIGGER):
        query = msg[len(WIKI_TRIGGER):].strip()
        if not query:
            return "Tu dois me dire ce que tu veux que je cherche sur Wikipédia."
        try:
            res = recherche_wikipedia(query)
            return chatbot_reponse(f"Voici ce que j'ai trouvé sur Wikipédia :\n{res}" if res else "Désolé, rien trouvé de pertinent sur Wikipédia.")
        except Exception as e:
            return chatbot_reponse(f"Erreur lors de la recherche Wikipédia : {e}")

    if msg.lower().startswith(GOOGLE_TRIGGER):
        query = msg[len(GOOGLE_TRIGGER):].strip()
        if not query:
            return "Tu dois me dire ce que tu veux que je cherche sur Google."
        try:
            res = recherche_google(query)
            return chatbot_reponse(f"Voici ce que j'ai trouvé via Google :\n{res}" if res else "Désolé, rien trouvé de pertinent via Google.")
        except Exception as e:
            return chatbot_reponse(f"Erreur lors de la recherche Google : {e}")

    # 🤖 GPT-2 pour tout le reste
    prompt = f"Utilisateur : {msg}\nAssistant :"
    gpt2_output = generate_gpt2_response(prompt)
    return chatbot_reponse(gpt2_output)
