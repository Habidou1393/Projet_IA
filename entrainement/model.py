import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    Attention multi-tête causale (masquée) pour empêcher la fuite d'information future.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd doit être divisible par n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        # Mask causal (lower-triangular), bool pour efficacité
        mask = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        B, T, C = x.size()
        # Calcul des clés, requêtes, valeurs pour chaque tête
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Produit scalaire attention
        att = torch.matmul(q, k.transpose(-2, -1))  # (B, nh, T, T)
        att = att / math.sqrt(self.head_dim)

        # Application du masque causal
        att = att.masked_fill(~self.mask[:T, :T], float('-inf'))

        att = F.softmax(att, dim=-1)  # attention weights
        y = torch.matmul(att, v)  # (B, nh, T, hs)

        # Reshape et projection finale
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec attention causale suivie d'un feed-forward.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Mini GPT-like Transformer Language Model.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 64,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialisation des poids de la position
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        Args:
            idx: Tensor shape (B, T) contenant les indices des tokens d'entrée.
        Returns:
            logits: Tensor shape (B, T, vocab_size) contenant les logits de prédiction.
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        tok_emb = self.token_embed(idx)  # (B, T, n_embd)
        pos_emb = self.pos_embed[:, :T, :]  # (1, T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Génère des tokens à partir d'un prompt initial.

        Args:
            idx: Tensor (B, T) des tokens de contexte initiaux.
            max_new_tokens: Nombre maximum de tokens à générer.
            eos_token_id: ID du token fin de séquence pour arrêter la génération.
            temperature: Float > 0, contrôle la diversité de la génération.
            top_k: Optionnel, top-k sampling.

        Returns:
            Tensor (B, T + max_new_tokens) avec les tokens générés.
        """
        self.eval()
        B, T = idx.size()
        generated = idx

        for _ in range(max_new_tokens):
            # Troncature au block_size
            input_ids = generated[:, -self.block_size :]
            logits = self.forward(input_ids)  # (B, T', vocab_size)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_logits, _ = torch.topk(logits, top_k)
                min_topk = topk_logits[:, -1].unsqueeze(1)
                logits = torch.where(
                    logits < min_topk, torch.full_like(logits, float("-inf")), logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
