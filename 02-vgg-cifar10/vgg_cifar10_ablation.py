"""
Day 2: VGG-style CNN on CIFAR-10 - ablation of augmentation, BatchNorm and Dropout (PyTorch).

Script version of vgg_cifar10_ablation.ipynb. Runs five experiments (A to E),
saves figures, a results CSV and model weights to ./outputs (or /kaggle/working on Kaggle).
A GPU is strongly recommended (about 15-30 minutes on a Kaggle T4).
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background: from AlexNet to VGG
# ======================================================================


import copy, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
VAL_SIZE = 5000
ARCH = "vgg6"          # try "vgg11" as an extension

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

print("PyTorch:", torch.__version__, "| Device:", DEVICE)


# ======================================================================
# 2. Data Pipeline
# ======================================================================
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
normalize = transforms.Normalize(mean, std)

tf_plain = transforms.Compose([transforms.ToTensor(), normalize])
tf_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

full_plain = datasets.CIFAR10("./data", train=True, download=True, transform=tf_plain)
full_aug   = datasets.CIFAR10("./data", train=True, download=False, transform=tf_aug)
test_ds    = datasets.CIFAR10("./data", train=False, download=True, transform=tf_plain)
classes = full_plain.classes

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(full_plain), generator=g).tolist()
val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

train_plain = Subset(full_plain, train_idx)   # no augmentation
train_aug   = Subset(full_aug,   train_idx)   # with augmentation
val_ds      = Subset(full_plain, val_idx)

def make_loader(ds, shuffle=False, bs=BATCH_SIZE):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True)

val_dl, test_dl = make_loader(val_ds, bs=512), make_loader(test_ds, bs=512)
train_eval_dl = make_loader(train_plain, bs=512)   # clean pass over train data for the final gap

print(f"train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_ds):,}")
print("classes:", classes)

mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)
def denorm(x):
    return (x * std_t + mean_t).clamp(0, 1)

# Samples + what augmentation does to a single image
raw = datasets.CIFAR10("./data", train=True, download=False)
aug_demo = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
for i in range(8):
    img, lab = raw[i]
    axes[0, i].imshow(img); axes[0, i].set_title(classes[lab], fontsize=9); axes[0, i].axis("off")
img, lab = raw[0]
axes[1, 0].imshow(img); axes[1, 0].set_title("original", fontsize=9); axes[1, 0].axis("off")
for i in range(1, 8):
    axes[1, i].imshow(aug_demo(img)); axes[1, i].set_title("augmented", fontsize=9); axes[1, i].axis("off")
plt.suptitle("Top: CIFAR-10 samples | Bottom: random augmentations of one image (crop + flip)")
plt.tight_layout()
save_fig("data_and_augmentation.png")
plt.show()


# ======================================================================
# 3. Model Implementation
# ======================================================================
CFGS = {
    "vgg6":  [64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}

class VGGNet(nn.Module):
    def __init__(self, arch="vgg6", batch_norm=False, dropout=0.0, num_classes=10):
        super().__init__()
        layers, in_ch, n_pool = [], 3, 0
        for v in CFGS[arch]:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2)); n_pool += 1
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=not batch_norm))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        self.features = nn.Sequential(*layers)
        spatial = 32 // (2 ** n_pool)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * spatial * spatial, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())

m = VGGNet(ARCH)
print(m)
print(f"\nParameters (no BN, no dropout): {count_params(m):,}")
if ARCH == "vgg6":
    assert count_params(m) == 3_248_202, "unexpected parameter count"
print(f"Parameters (with BN): {count_params(VGGNet(ARCH, batch_norm=True)):,}")

# Shape trace
x = torch.randn(1, 3, 32, 32)
for layer in m.features:
    x = layer(x)
    if isinstance(layer, (nn.MaxPool2d,)):
        print(f"{layer.__class__.__name__:<10} -> {tuple(x.shape)}")
print("classifier ->", tuple(m.classifier(x).shape))


# ======================================================================
# 4. Training Setup
# ======================================================================
use_amp = DEVICE.type == "cuda"

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            n          += x.size(0)
    return total_loss / n, correct / n


def train_experiment(name, augment, batch_norm, dropout):
    print(f"\n=== {name} | aug={augment} bn={batch_norm} dropout={dropout} ===")
    torch.manual_seed(SEED)
    model = VGGNet(ARCH, batch_norm=batch_norm, dropout=dropout).to(DEVICE)
    train_dl = make_loader(train_aug if augment else train_plain, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion)
        scheduler.step()
        for k, v in zip(hist, [tr_loss, va_loss, tr_acc, va_acc]):
            hist[k].append(v)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.4f}")

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    _, test_acc = run_epoch(model, test_dl, criterion)
    _, train_acc_clean = run_epoch(model, train_eval_dl, criterion)
    print(f"-> best epoch {best_epoch} | val {best_val:.4f} | test {test_acc:.4f} "
          f"| clean train {train_acc_clean:.4f} | {elapsed:.0f}s")
    return dict(model=model, hist=hist, best_epoch=best_epoch, best_val_acc=best_val,
                test_acc=test_acc, train_acc_clean=train_acc_clean,
                params=count_params(model), time=elapsed)

results = {}


# Experiment A: baseline
results["A: baseline"] = train_experiment("A: baseline", augment=False, batch_norm=False, dropout=0.0)


# Experiment B: +augmentation
results["B: +augmentation"] = train_experiment("B: +augmentation", augment=True, batch_norm=False, dropout=0.0)


# Experiment C: +batchnorm
results["C: +batchnorm"] = train_experiment("C: +batchnorm", augment=False, batch_norm=True, dropout=0.0)


# Experiment D: aug+batchnorm
results["D: aug+batchnorm"] = train_experiment("D: aug+batchnorm", augment=True, batch_norm=True, dropout=0.0)


# Experiment E: aug+bn+dropout
results["E: aug+bn+dropout"] = train_experiment("E: aug+bn+dropout", augment=True, batch_norm=True, dropout=0.5)


# ======================================================================
# 5. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Params": f"{r['params']:,}",
        "Best epoch": r["best_epoch"],
        "Val acc (%)": round(r["best_val_acc"] * 100, 2),
        "Test acc (%)": round(r["test_acc"] * 100, 2),
        "Train acc, clean (%)": round(r["train_acc_clean"] * 100, 2),
        "Gap train-test (pp)": round((r["train_acc_clean"] - r["test_acc"]) * 100, 2),
        "Time (s)": round(r["time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(summary.to_string(index=False))

# Training curves for every experiment
names = list(results)
ep = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7))
for j, name in enumerate(names):
    h = results[name]["hist"]
    axes[0, j].plot(ep, np.array(h["train_acc"]) * 100, "--", label="train")
    axes[0, j].plot(ep, np.array(h["val_acc"]) * 100, label="val")
    axes[0, j].set_title(name); axes[0, j].set_ylim(30, 101); axes[0, j].set_xlabel("epoch")
    axes[1, j].plot(ep, h["train_loss"], "--", label="train")
    axes[1, j].plot(ep, h["val_loss"], label="val")
    axes[1, j].set_xlabel("epoch")
    if j == 0:
        axes[0, j].set_ylabel("accuracy (%)"); axes[1, j].set_ylabel("loss")
        axes[0, j].legend(); axes[1, j].legend()
plt.suptitle("Training curves (train accuracy is measured on augmented data / with dropout when those are on)")
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# Overfitting analysis + head-to-head comparison
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
for name in names:
    h = results[name]["hist"]
    ax[0].plot(ep, np.array(h["val_acc"]) * 100, label=name)
    gap_curve = np.array(h["val_loss"]) - np.array(h["train_loss"])
    ax[1].plot(ep, gap_curve, label=name)
ax[0].set_title("Validation accuracy (%)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].axhline(0, color="gray", lw=0.8, ls="--")
ax[1].set_title("Loss gap per epoch (val - train)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)

gaps = [(r["train_acc_clean"] - r["test_acc"]) * 100 for r in results.values()]
bars = ax[2].bar(range(len(names)), gaps, color="tab:red", alpha=0.75)
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels([n.split(":")[0] for n in names])
ax[2].set_title("Final generalization gap: clean train - test (pp)")
for b, g_ in zip(bars, gaps):
    ax[2].text(b.get_x() + b.get_width() / 2, g_ + 0.2, f"{g_:.1f}", ha="center", fontsize=9)
plt.tight_layout()
save_fig("ablation_summary.png")
plt.show()

# Select the best configuration by VALIDATION accuracy (not test) and analyze it
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best_model = results[best_name]["model"]
print("Best config by validation accuracy:", best_name,
      f"| test acc {results[best_name]['test_acc']*100:.2f}%")

@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, L, X = [], [], []
    for x, y in loader:
        P.append(model(x.to(DEVICE)).argmax(1).cpu()); L.append(y); X.append(x)
    return torch.cat(P).numpy(), torch.cat(L).numpy(), torch.cat(X)

preds, labels, imgs = predict(best_model, test_dl)
cm = confusion_matrix(labels, preds)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=classes, yticklabels=classes)
ax[0].set_title(f"Confusion matrix - {best_name}")
cm_err = cm.copy(); np.fill_diagonal(cm_err, 0)
sns.heatmap(cm_err, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax[1],
            xticklabels=classes, yticklabels=classes)
ax[1].set_title("Errors only (diagonal removed)")
for a in ax:
    a.set_xlabel("Predicted"); a.set_ylabel("True")
plt.tight_layout()
save_fig("confusion_matrix.png")
plt.show()

print(classification_report(labels, preds, target_names=classes, digits=3))

# Per-class accuracy + most confused pairs
per_class = cm.diagonal() / cm.sum(1)
order = np.argsort(per_class)
plt.figure(figsize=(8, 4))
plt.barh([classes[i] for i in order], per_class[order] * 100, color="tab:blue")
plt.xlabel("accuracy (%)"); plt.title(f"Per-class test accuracy - {best_name}")
plt.xlim(max(0, per_class.min() * 100 - 10), 100)
plt.tight_layout()
save_fig("per_class_accuracy.png")
plt.show()

top = np.dstack(np.unravel_index(np.argsort(cm_err.ravel())[::-1][:5], cm.shape))[0]
print("Top-5 confusions (true -> predicted : count)")
for t, p in top:
    print(f"  {classes[t]} -> {classes[p]} : {cm_err[t, p]}")

# Misclassified test images
wrong = np.where(preds != labels)[0][:16]
fig, axes = plt.subplots(2, 8, figsize=(14, 4.2))
for a, i in zip(axes.ravel(), wrong):
    a.imshow(denorm(imgs[i]).permute(1, 2, 0))
    a.set_title(f"T: {classes[labels[i]]}\nP: {classes[preds[i]]}", fontsize=8, color="red")
    a.axis("off")
plt.suptitle("Misclassified test images (T = true, P = predicted)")
plt.tight_layout()
save_fig("misclassified.png")
plt.show()


# ======================================================================
# 6. What Did the First Layer Learn?
# ======================================================================
w = best_model.features[0].weight.detach().cpu()        # (64, 3, 3, 3)
w = (w - w.amin(dim=(1, 2, 3), keepdim=True)) / (w.amax(dim=(1, 2, 3), keepdim=True) - w.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for a, f in zip(axes.ravel(), w):
    a.imshow(f.permute(1, 2, 0)); a.axis("off")
plt.suptitle(f"Learned first-layer filters (3x3 RGB) - {best_name}")
plt.tight_layout()
save_fig("conv1_filters.png")
plt.show()

# Save weights of every experiment
for name, r in results.items():
    fname = "vgg_" + name.split(":")[0].lower() + ".pth"
    torch.save(r["model"].state_dict(), OUT / fname)
print("Saved to", OUT)


# ======================================================================
# 7. Analysis
# ======================================================================

# ======================================================================
# Key findings
# ======================================================================

# ======================================================================
# Limitations
# ======================================================================

# ======================================================================
# Possible improvements
# ======================================================================

# ======================================================================
# Next in this series
# ======================================================================

# ======================================================================
# References
# ======================================================================