import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizers import ByteLevelBPETokenizer

# Hyperparamètres
BATCH_SIZE = 16
BLOCK_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4

# Chargement du tokenizer entraîné
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json", "tokenizer/merges.txt"
)
VOCAB_SIZE = tokenizer.get_vocab_size()

class TextDataset(Dataset):
    def __init__(self, filepath, block_size, tokenizer):
        with open(filepath, encoding="utf-8") as f:
            data = f.read()
        self.tokens = tokenizer.encode(data).ids
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return torch.tensor(x), torch.tensor(y)

def train():
    dataset = TextDataset("data/corpus_formate.txt", BLOCK_SIZE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniGPT(VOCAB_SIZE, block_size=BLOCK_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)  # logits shape: (batch_size, seq_len, vocab_size)
            logits = logits.view(-1, VOCAB_SIZE)
            targets = y.view(-1)

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} / {EPOCHS} — loss moyenne: {total_loss / len(dataloader):.4f}")

    # Sauvegarder le modèle, création du dossier s'il n'existe pas
    save_path = "entrainement/model-output/minigpt.pth"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans '{save_path}'")

if __name__ == "__main__":
    train()
