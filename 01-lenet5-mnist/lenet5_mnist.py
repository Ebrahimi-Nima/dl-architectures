"""
Day 1: LeNet-5 on MNIST (PyTorch)
Works locally and on Kaggle (turn on GPU: Settings -> Accelerator).
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- Config ----------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)


# ---------------- Model ----------------
class LeNet5(nn.Module):
    """
    Input : 1 x 32 x 32  (MNIST 28x28 padded by 2 on each side)
    C1    : Conv 1->6,  k=5  -> 6 x 28 x 28
    S2    : AvgPool 2x2      -> 6 x 14 x 14
    C3    : Conv 6->16, k=5  -> 16 x 10 x 10
    S4    : AvgPool 2x2      -> 16 x 5 x 5
    C5    : Conv 16->120,k=5 -> 120 x 1 x 1
    F6    : FC 120->84
    Out   : FC 84->10
    activation: tanh (as in the original paper); set use_relu_maxpool=True for the modern variant
    """

    def __init__(self, num_classes=10, use_relu_maxpool=False):
        super().__init__()
        act = nn.ReLU if use_relu_maxpool else nn.Tanh
        pool = nn.MaxPool2d if use_relu_maxpool else nn.AvgPool2d

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), act(),
            pool(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), act(),
            pool(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5), act(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84), act(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- Data ----------------
tfm = transforms.Compose([
    transforms.Pad(2),                      # 28x28 -> 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512)


# ---------------- Train / Eval ----------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train(use_relu_maxpool=False):
    model = LeNet5(use_relu_maxpool=use_relu_maxpool).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    hist = {"tr_loss": [], "te_loss": [], "tr_acc": [], "te_acc": []}

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
        te_loss, te_acc = run_epoch(model, test_dl, criterion)
        for k, v in zip(hist, [tr_loss, te_loss, tr_acc, te_acc]):
            hist[k].append(v)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f} | {time.time()-t0:.1f}s")
    return model, hist


# ---------------- Plots ----------------
def plot_history(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["te_loss"], label="test")
    ax[0].set_title(f"{name} - Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["tr_acc"], label="train"); ax[1].plot(hist["te_acc"], label="test")
    ax[1].set_title(f"{name} - Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{name}_curves.png", dpi=150); plt.show()


def plot_confusion(model, name):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            labels += y.tolist()
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", colorbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"{name}_confusion.png", dpi=150); plt.show()


if __name__ == "__main__":
    m = LeNet5()
    print(m)
    print("Total params:", sum(p.numel() for p in m.parameters()))  # expected: 61,706

    # Experiment A: original (tanh + avgpool)
    model_a, hist_a = train(use_relu_maxpool=False)
    plot_history(hist_a, "lenet5_tanh_avgpool")
    plot_confusion(model_a, "lenet5_tanh_avgpool")
    torch.save(model_a.state_dict(), "lenet5_tanh_avgpool.pth")

    # Experiment B: modern variant (ReLU + maxpool) for comparison
    model_b, hist_b = train(use_relu_maxpool=True)
    plot_history(hist_b, "lenet5_relu_maxpool")
    plot_confusion(model_b, "lenet5_relu_maxpool")
    torch.save(model_b.state_dict(), "lenet5_relu_maxpool.pth")
