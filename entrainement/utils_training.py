import torch
import os

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    vocab_size,
    loss_fn=torch.nn.functional.cross_entropy,
    scheduler=None,
    log_every=100
):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # (batch_size, seq_len, vocab_size)
        logits = logits.view(-1, vocab_size)
        targets = y.view(-1)

        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_tokens = targets.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if batch_idx % log_every == 0:
            print(f"[Batch {batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_tokens
    return avg_loss


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé dans '{path}'")


def load_model(model, path, device):
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Modèle chargé depuis '{path}'")
    else:
        print(f"Attention : modèle non trouvé à '{path}'")
