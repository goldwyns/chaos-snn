import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.chaos_snn import ChaosSNNSeizureDetector


def generate_dummy_batch(batch_size, n_channels, T):
    """
    Create fake EEG-like data & labels.
    """
    x = torch.randn(batch_size, n_channels, T)
    # Make "seizure" samples slightly higher amplitude
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
    x = x + y.view(-1, 1, 1) * 0.5
    return x, y


def train_dummy(
    n_channels=16,
    T=128,
    batch_size=32,
    n_epochs=5,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = ChaosSNNSeizureDetector(
        n_channels=n_channels,
        n_hidden=64,
        alpha=3.5,
        beta=0.2,
        gamma=0.5,
        lam=0.7,
        v_th=1.0,
        alpha_rec=1.0,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for _ in tqdm(range(100), desc=f"Epoch {epoch}"):
            x, y = generate_dummy_batch(batch_size, n_channels, T)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            p, extra = model(x)
            loss = criterion(p, y)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / 100.0
        print(f"Epoch {epoch} - loss: {avg_loss:.4f}")

    print("✅ Dummy training finished.")


if __name__ == "__main__":
    train_dummy()
