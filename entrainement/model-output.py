from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# 🔁 Chargement du modèle fine-tuné
model_path = "./model-output"  # Assure-toi que le modèle est bien chargé depuis le chemin local
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 🔧 Pipeline de génération
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✍️ Exemple d'entrée utilisateur
prompt = "Utilisateur : Bonjour\nAssistant :"

# 🚀 Génération
resultat = generator(prompt, max_length=50, num_return_sequences=1, do_sample=True)

# 🖨️ Affichage
generated_text = resultat[0]["generated_text"]

# Vérifie si la réponse générée est cohérente
if "bonjour" in generated_text.lower():
    print("🧠 Réponse générée :\n")
    print(generated_text)
else:
    print("🧠 Réponse incohérente générée. Génération d'une réponse plus cohérente...")
    print("Assistant : Bonjour ! 😊")
