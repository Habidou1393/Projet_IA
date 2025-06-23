import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from torch import amp  # nouvelle API pour amp

# --- Classes essentielles simplifiées pour que ce soit fonctionnel ---

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)    # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MediumGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=256, n_layer=4, n_head=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length longer than block size"
        tok_emb = self.token_emb(idx)
        x = tok_emb + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class TextDataset(Dataset):
    def __init__(self, file_path, block_size, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = tokenizer.encode(text)
        self.tokens = encoded.ids
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    BATCH_SIZE = 8
    BLOCK_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 3e-4

    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    VOCAB_SIZE = tokenizer.get_vocab_size()

    dataset = TextDataset("data/corpus_formate.txt", BLOCK_SIZE, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé pour l'entraînement : {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=4 if device == "cuda" else 0
    )

    model = MediumGPT(VOCAB_SIZE, block_size=BLOCK_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = amp.GradScaler()  # <-- correction ici : pas d'argument device_type

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with amp.autocast(device_type=device):  # device est 'cuda' ou 'cpu'
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} / {EPOCHS} — perte moyenne: {total_loss / len(dataloader):.4f}")

    save_path = "entrainement/model-output/mediumgpt.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé dans '{save_path}'")

if __name__ == "__main__":
    train()
