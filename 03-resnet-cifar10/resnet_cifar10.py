"""
Day 3: ResNet from scratch on CIFAR-10 - plain vs residual networks (PyTorch).

Script version of resnet_cifar10.ipynb. Trains Plain-20, ResNet-20, Plain-56, ResNet-56 and ResNet-18,
saves figures, a results CSV and model weights to ./outputs (or /kaggle/working on Kaggle).
A GPU is strongly recommended (about 30-45 minutes on a Kaggle T4).
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
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

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

# Reference result from Day 2 (best VGG-style model: augmentation + BatchNorm), same split and protocol
DAY2 = {"name": "Day 2: VGG-6 (aug+BN)", "params": 3_249_098, "test_acc": 88.93, "time": 225}

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

train_plain = Subset(full_plain, train_idx)   # clean (for the final train-accuracy measurement)
train_aug   = Subset(full_aug,   train_idx)   # augmented (used for training)
val_ds      = Subset(full_plain, val_idx)

def make_loader(ds, shuffle=False, bs=BATCH_SIZE):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True)

train_dl = make_loader(train_aug, shuffle=True)
val_dl, test_dl = make_loader(val_ds, bs=512), make_loader(test_ds, bs=512)
train_eval_dl = make_loader(train_plain, bs=512)

mean_t = torch.tensor(mean).view(3, 1, 1)
std_t  = torch.tensor(std).view(3, 1, 1)
def denorm(x):
    return (x * std_t + mean_t).clamp(0, 1)

print(f"train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_ds):,}")


# ======================================================================
# 3. Model Implementation
# ======================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, skip):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = None
        if skip:
            if stride == 1 and in_ch == out_ch:
                self.shortcut = nn.Identity()
            else:   # projection shortcut (1x1 conv + BN) when shape changes
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)          # y = F(x) + x
        return self.relu(out)


class CifarNet(nn.Module):
    def __init__(self, blocks, widths, skip=True, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, in_ch = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks, widths)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(in_ch, w, stride, skip))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())

def depth_of(blocks):
    return 1 + 2 * sum(blocks) + 1

R18 = dict(blocks=[2, 2, 2, 2], widths=(64, 128, 256, 512))
m = CifarNet(**R18, skip=True)
print(f"ResNet-18 depth: {depth_of(R18['blocks'])} | parameters: {count_params(m):,}")
assert count_params(m) == 11_173_962, "unexpected parameter count for ResNet-18"

for name, blocks in [("20", [3, 3, 3]), ("56", [9, 9, 9])]:
    p_plain = count_params(CifarNet(blocks, (16, 32, 64), skip=False))
    p_res   = count_params(CifarNet(blocks, (16, 32, 64), skip=True))
    print(f"CIFAR-{name}: plain {p_plain:,} | residual {p_res:,} (+{p_res - p_plain:,} from 1x1 projection shortcuts)")

# Shape trace for ResNet-18 (prints whenever the feature-map shape changes)
x = torch.randn(1, 3, 32, 32)
x = m.stem(x); print("stem   ->", tuple(x.shape))
for i, blk in enumerate(m.layers):
    prev = x.shape
    x = blk(x)
    if x.shape != prev:
        print(f"block {i} ->", tuple(x.shape))
print("fc     ->", tuple(m.fc(torch.flatten(m.pool(x), 1)).shape))


# ======================================================================
# 4. Gradient Flow at Initialization
# ======================================================================
def grad_norms(blocks, widths, skip):
    torch.manual_seed(SEED)
    model = CifarNet(blocks, widths, skip=skip).to(DEVICE).train()
    x, y = next(iter(make_loader(train_plain, shuffle=False, bs=256)))
    loss = nn.CrossEntropyLoss()(model(x.to(DEVICE)), y.to(DEVICE))
    loss.backward()
    return [mod.weight.grad.norm().item() for mod in model.modules()
            if isinstance(mod, nn.Conv2d) and mod.kernel_size == (3, 3)]

g_plain = grad_norms([9, 9, 9], (16, 32, 64), skip=False)
g_res   = grad_norms([9, 9, 9], (16, 32, 64), skip=True)

plt.figure(figsize=(8, 4))
plt.semilogy(range(1, len(g_plain) + 1), g_plain, label="Plain-56")
plt.semilogy(range(1, len(g_res) + 1),   g_res,   label="ResNet-56")
plt.xlabel("3x3 conv layer index (1 = closest to the input)")
plt.ylabel("gradient norm (log scale)")
plt.title("Gradient norm per layer at initialization (one batch)")
plt.legend(); plt.grid(alpha=0.3)
save_fig("gradient_flow.png")
plt.show()

for n, g_ in [("Plain-56", g_plain), ("ResNet-56", g_res)]:
    print(f"{n}: first conv {g_[0]:.2e} | last conv {g_[-1]:.2e} | ratio first/last {g_[0] / g_[-1]:.3f}")


# ======================================================================
# 5. Training Setup
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


def train_experiment(name, blocks, widths, skip):
    print(f"\n=== {name} | depth {depth_of(blocks)} | skip={skip} ===")
    torch.manual_seed(SEED)
    model = CifarNet(blocks, widths, skip=skip).to(DEVICE)
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
    return dict(model=model, hist=hist, depth=depth_of(blocks), skip=skip, best_epoch=best_epoch,
                best_val_acc=best_val, test_acc=test_acc, train_acc_clean=train_acc_clean,
                params=count_params(model), time=elapsed)

results = {}


# Experiment A: Plain-20
results["A: Plain-20"] = train_experiment("A: Plain-20", blocks=[3, 3, 3], widths=(16, 32, 64), skip=False)


# Experiment B: ResNet-20
results["B: ResNet-20"] = train_experiment("B: ResNet-20", blocks=[3, 3, 3], widths=(16, 32, 64), skip=True)


# Experiment C: Plain-56
results["C: Plain-56"] = train_experiment("C: Plain-56", blocks=[9, 9, 9], widths=(16, 32, 64), skip=False)


# Experiment D: ResNet-56
results["D: ResNet-56"] = train_experiment("D: ResNet-56", blocks=[9, 9, 9], widths=(16, 32, 64), skip=True)


# Experiment E: ResNet-18
results["E: ResNet-18"] = train_experiment("E: ResNet-18", blocks=[2, 2, 2, 2], widths=(64, 128, 256, 512), skip=True)


# ======================================================================
# 6. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Depth": r["depth"],
        "Skip": "yes" if r["skip"] else "no",
        "Params": f"{r['params']:,}",
        "Best epoch": r["best_epoch"],
        "Val acc (%)": round(r["best_val_acc"] * 100, 2),
        "Test acc (%)": round(r["test_acc"] * 100, 2),
        "Train acc, clean (%)": round(r["train_acc_clean"] * 100, 2),
        "Gap (pp)": round((r["train_acc_clean"] - r["test_acc"]) * 100, 2),
        "Final train loss": round(r["hist"]["train_loss"][-1], 3),
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
    axes[0, j].plot(ep, np.array(h["train_acc"]) * 100, "--", label="train (augmented)")
    axes[0, j].plot(ep, np.array(h["val_acc"]) * 100, label="val")
    axes[0, j].set_title(name); axes[0, j].set_ylim(30, 101); axes[0, j].set_xlabel("epoch")
    axes[1, j].plot(ep, h["train_loss"], "--", label="train")
    axes[1, j].plot(ep, h["val_loss"], label="val")
    axes[1, j].set_xlabel("epoch")
    if j == 0:
        axes[0, j].set_ylabel("accuracy (%)"); axes[1, j].set_ylabel("loss")
        axes[0, j].legend(); axes[1, j].legend()
plt.suptitle("Training curves (training metrics are computed on augmented batches)")
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))

for pair, a, title in [(("A: Plain-20", "C: Plain-56"), ax[0], "Plain networks: training loss"),
                       (("B: ResNet-20", "D: ResNet-56"), ax[1], "Residual networks: training loss")]:
    for nm in pair:
        a.plot(ep, results[nm]["hist"]["train_loss"], label=f"{nm} (final {results[nm]['hist']['train_loss'][-1]:.3f})")
    a.set_title(title); a.set_xlabel("epoch"); a.set_ylabel("training loss (augmented)")
    a.legend(); a.grid(alpha=0.3)

depth_labels = ["depth 20", "depth 56"]
plain_acc = [results["A: Plain-20"]["test_acc"] * 100, results["C: Plain-56"]["test_acc"] * 100]
res_acc   = [results["B: ResNet-20"]["test_acc"] * 100, results["D: ResNet-56"]["test_acc"] * 100]
xs, w = np.arange(2), 0.35
b1 = ax[2].bar(xs - w / 2, plain_acc, w, label="Plain", color="tab:gray")
b2 = ax[2].bar(xs + w / 2, res_acc,   w, label="Residual", color="tab:blue")
for bars in (b1, b2):
    for b in bars:
        ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}", ha="center", fontsize=9)
ax[2].set_xticks(xs); ax[2].set_xticklabels(depth_labels)
ax[2].set_ylim(min(plain_acc + res_acc) - 4, max(plain_acc + res_acc) + 2)
ax[2].set_title("Test accuracy (%) by depth"); ax[2].legend()

plt.tight_layout()
save_fig("degradation.png")
plt.show()

# Efficiency view: accuracy vs parameters (marker area ~ training time)
plt.figure(figsize=(8, 5))
for name, r in results.items():
    plt.scatter(r["params"], r["test_acc"] * 100, s=r["time"] * 1.2,
                marker="o" if r["skip"] else "s", alpha=0.75, label=name)
    plt.annotate(name.split(":")[1].strip(), (r["params"], r["test_acc"] * 100),
                 textcoords="offset points", xytext=(6, 6), fontsize=9)
plt.scatter(DAY2["params"], DAY2["test_acc"], s=DAY2["time"] * 1.2, facecolors="none",
            edgecolors="black", linewidths=1.5, label=DAY2["name"])
plt.xscale("log"); plt.xlabel("parameters (log scale)"); plt.ylabel("test accuracy (%)")
plt.title("Accuracy vs model size (circle = residual, square = plain; area ~ training time)")
plt.legend(fontsize=8); plt.grid(alpha=0.3)
save_fig("accuracy_vs_params.png")
plt.show()


# ======================================================================
# Error Analysis (best configuration)
# ======================================================================
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best_model = results[best_name]["model"]
print("Best config by validation accuracy:", best_name,
      f"| test acc {results[best_name]['test_acc']*100:.2f}%")
print(f"Day 2 reference ({DAY2['name']}): {DAY2['test_acc']:.2f}% test acc, {DAY2['params']:,} params")

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0], xticklabels=classes, yticklabels=classes)
ax[0].set_title(f"Confusion matrix - {best_name}")
cm_err = cm.copy(); np.fill_diagonal(cm_err, 0)
sns.heatmap(cm_err, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax[1], xticklabels=classes, yticklabels=classes)
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

# Save weights of every experiment
for name, r in results.items():
    fname = "resnet_" + name.split(":")[0].lower() + ".pth"
    torch.save(r["model"].state_dict(), OUT / fname)
print("Saved to", OUT)


# ======================================================================
# 7. Analysis
# ======================================================================