import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import math

# ----- Définition des blocs Transformer et MediumGPT -----

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MediumGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=12, n_layer=8, block_size=128):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed[:, :T, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ----- Dataset -----

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

# ----- Entraînement -----

def train():
    # Hyperparamètres
    BATCH_SIZE = 8
    BLOCK_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 3e-4

    # Chargement tokenizer entraîné
    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    VOCAB_SIZE = tokenizer.get_vocab_size()

    dataset = TextDataset("data/corpus_formate.txt", BLOCK_SIZE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entraînement sur {device}")

    model = MediumGPT(VOCAB_SIZE, block_size=BLOCK_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)  # (B, T, vocab_size)
            logits = logits.view(-1, VOCAB_SIZE)
            targets = y.view(-1)

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} / {EPOCHS} — perte moyenne: {total_loss / len(dataloader):.4f}")

    # Sauvegarder le modèle
    save_path = "entrainement/model-output/mediumgpt.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans '{save_path}'")

if __name__ == "__main__":
    train()
