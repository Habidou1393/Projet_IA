import os
import sys
import logging
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 📁 Chemin vers le corpus ===
corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mon_corpus.txt'))

if not os.path.isfile(corpus_path):
    logger.error(f"Le fichier corpus est introuvable : {corpus_path}")
    sys.exit(1)
if os.path.getsize(corpus_path) == 0:
    logger.error(f"Le fichier corpus est vide : {corpus_path}")
    sys.exit(1)

logger.info(f"Chargement du corpus depuis : {corpus_path}")
dataset = load_dataset("text", data_files={"train": corpus_path}, split="train")

# === 🄤 Tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# === Tokenisation avec gestion de la longueur de séquence ===
logger.info("Tokenisation...")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# === Découpage en blocs avec "Sliding Window" ===
def sliding_window_tokenize(examples, block_size=1024):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    # Créer des fenêtres glissantes
    result = {
        "input_ids": [concatenated[i:i + block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [[1] * block_size for _ in range(total_length // block_size)]  # Attention mask
    }
    return result

lm_dataset = tokenized_dataset.map(sliding_window_tokenize, batched=True)

if len(lm_dataset) == 0:
    logger.error("❌ Aucun exemple valide n'a été généré. Vérifie ton corpus.")
    sys.exit(1)

logger.info(f"Nombre d'exemples finaux : {len(lm_dataset)}")

# === 🤖 Modèle ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# === 📆 Préparation des batches ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === ⚙️ Configuration de l'entraînement avec hyperparamètres optimisés ===
training_args = TrainingArguments(
    output_dir="./model-output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,  # Ajuste le taux d'apprentissage
    lr_scheduler_type="linear",  # Utilisation du scheduler linéaire
    warmup_steps=500,  # Nombre d'étapes de warmup
    report_to="none",
)

# === 🚀 Entraîneur ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# === 💾 Sauvegarde finale ===
model_path = "./model-output"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
logger.info(f"✅ Modèle et tokenizer sauvegardés dans : {model_path}")
