import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Chemin vers ton corpus
corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mon_corpus.txt'))

# Chargement du dataset
dataset = load_dataset("text", data_files={"train": corpus_path}, split="train")

# Tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # important pour padding

# Tokenisation des données avec découpage en blocs de 64 tokens (pour éviter bloc vide)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Préparation du DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Chargement modèle GPT-2 pré-entraîné
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./model-output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=200,
    report_to="none"
)

# Création du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Démarrer l’entraînement
trainer.train()

# Sauvegarde finale du modèle et tokenizer
trainer.save_model("./model-output")
tokenizer.save_pretrained("./model-output")

print("✅ Fine-tuning terminé et modèle sauvegardé dans ./model-output")
