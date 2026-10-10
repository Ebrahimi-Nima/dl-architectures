"""
Day 4: EfficientNet-B0 transfer learning on plant disease images (PyTorch).

Script version of efficientnet_plant_disease.ipynb. Runs five experiments (frozen backbone, fine-tuning,
training from scratch, at 100% and 5% data) and saves figures, a results CSV and the best model.
Expects the 'New Plant Diseases Dataset' under /kaggle/input (it searches for train/ and valid/ folders).
A GPU and internet access (for the pretrained weights) are required; about 40-60 minutes on a Kaggle T4.
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
import os, copy, time, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report

SEED = 42
IMG = 224
BATCH_SIZE = 64
HEAD_LR = 1e-3          # new classifier head (and all weights when training from scratch)
BACKBONE_LR = 2e-4      # pretrained backbone during fine-tuning (discriminative learning rate)
WEIGHT_DECAY = 1e-4
MAX_TRAIN_IMAGES = None # e.g. 20000 to speed things up if GPU time is short
NUM_WORKERS = min(4, os.cpu_count() or 2)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

print("PyTorch:", torch.__version__, "| Device:", DEVICE, "| workers:", NUM_WORKERS)


# ======================================================================
# 2. Dataset
# ======================================================================
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_input(root="/kaggle/input", max_items=12):
    """Print what is mounted under /kaggle/input (two levels), to debug path problems."""
    root = Path(root)
    if not root.exists():
        print(root, "does not exist (are you running on Kaggle?)"); return
    items = sorted(root.iterdir())
    print(f"{root} contains {len(items)} item(s)")
    for p in items[:max_items]:
        print("  ", p)
        if p.is_dir():
            for q in sorted(p.iterdir())[:max_items]:
                print("      ", q)

def has_images(d):
    try:
        return any(Path(f).suffix.lower() in IMG_EXT for f in os.listdir(d))
    except OSError:
        return False

def is_class_root(path, dirs):
    """A folder whose sub-folders look like classes (they directly contain images)."""
    if len(dirs) < 2:
        return False
    probe = dirs[:3]
    return sum(has_images(Path(path) / d) for d in probe) >= min(2, len(probe))

def locate_data(root="/kaggle/input"):
    """Find (train_dir, valid_dir). valid_dir is None if no train/valid pair exists."""
    cands = [Path(p) for p, dirs, files in os.walk(root) if is_class_root(p, dirs)]
    if not cands:
        list_input(root)
        raise FileNotFoundError(
            "No image-folder dataset found under /kaggle/input (see the listing above). "
            "If it is empty, add the dataset with '+ Add Input' in the right panel and re-run.")
    train_dir = next((c for c in cands if c.name.lower() == "train"), None)
    valid_dir = next((c for c in cands if c.name.lower() in ("valid", "val", "validation")
                      and (train_dir is None or c.parent == train_dir.parent)), None)
    if train_dir is not None and valid_dir is not None:
        return train_dir, valid_dir
    return max(cands, key=lambda c: len(os.listdir(c))), None

list_input()
train_dir, valid_dir = locate_data()
print("train folder:", train_dir)
print("valid folder:", valid_dir)

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

tf_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    normalize,
])
tf_eval = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), normalize])

ds_aug  = datasets.ImageFolder(train_dir, transform=tf_train)   # training (augmented)
ds_eval = datasets.ImageFolder(train_dir, transform=tf_eval)    # clean version of the same files
classes = ds_eval.classes
NUM_CLASSES = len(classes)
targets = np.array(ds_eval.targets)
all_idx = np.arange(len(ds_eval))

if valid_dir is not None:
    train_idx, val_idx = train_test_split(all_idx, test_size=0.10, stratify=targets, random_state=SEED)
    test_ds = datasets.ImageFolder(valid_dir, transform=tf_eval)
    assert test_ds.classes == classes, "train and valid folders have different classes"
else:
    tv_idx, test_idx = train_test_split(all_idx, test_size=0.15, stratify=targets, random_state=SEED)
    train_idx, val_idx = train_test_split(tv_idx, test_size=0.15 / 0.85, stratify=targets[tv_idx], random_state=SEED)
    test_ds = Subset(ds_eval, test_idx)

if MAX_TRAIN_IMAGES and len(train_idx) > MAX_TRAIN_IMAGES:
    train_idx = train_test_split(train_idx, train_size=MAX_TRAIN_IMAGES, stratify=targets[train_idx], random_state=SEED)[0]

def make_loader(ds, shuffle=False, bs=BATCH_SIZE):
    # No persistent workers: a fresh set of workers is created for every pass over the data.
    # This is slightly slower to start but much more robust (an interrupted or crashed pass cannot
    # leave dead workers behind). If workers still crash, lower NUM_WORKERS (2, then 0) in the config cell.
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)

val_dl  = make_loader(Subset(ds_eval, val_idx), bs=64)
test_dl = make_loader(test_ds, bs=64)

print(f"classes: {NUM_CLASSES} | train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_ds):,}")

def short(c):
    """Compact label like 'Tomato/Late blight'."""
    if "___" in c:
        crop, disease = c.split("___", 1)
        return f"{crop.split('_')[0][:9]}/{disease.replace('_', ' ')[:16]}"
    return c[:24]

def denorm(x):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1); s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1)

# Class balance + sample images
counts = np.bincount(targets[train_idx], minlength=NUM_CLASSES)
rng = np.random.RandomState(SEED)
pick = rng.choice(len(ds_eval), 8, replace=False)

plt.figure(figsize=(14, 4))
plt.bar(range(NUM_CLASSES), counts, color="tab:green")
plt.xticks(range(NUM_CLASSES), [short(c) for c in classes], rotation=90, fontsize=7)
plt.title(f"Training images per class (min {counts.min():,}, max {counts.max():,})")
plt.tight_layout()
save_fig("class_distribution.png")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for a, i in zip(axes.ravel(), pick):
    img, lab = ds_eval[i]
    a.imshow(denorm(img).permute(1, 2, 0)); a.set_title(short(classes[lab]), fontsize=9); a.axis("off")
plt.suptitle("Random training samples")
plt.tight_layout()
save_fig("samples.png")
plt.show()


# ======================================================================
# 3. Models
# ======================================================================
class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduced):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, ch, 1)
    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)                 # squeeze: global average pool
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))     # excite: per-channel gate in (0, 1)
        return x * s

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand, k, stride):
        super().__init__()
        mid = in_ch * expand
        layers = []
        if expand != 1:                                       # 1) expand
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False),   # 2) depthwise
                   nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [SqueezeExcite(mid, max(1, in_ch // 4))]    # 3) squeeze-and-excitation
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]    # 4) project
        self.block = nn.Sequential(*layers)
        self.use_skip = stride == 1 and in_ch == out_ch       # 5) residual connection
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_skip else out

def count_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))

blk = MBConv(24, 40, expand=6, k=5, stride=2)
print("MBConv(24 -> 40, expand 6, k5, stride 2):", tuple(blk(torch.randn(1, 24, 56, 56)).shape),
      "| params:", f"{count_params(blk):,}")

standard  = nn.Conv2d(128, 128, 3, bias=False)
separable = nn.Sequential(nn.Conv2d(128, 128, 3, groups=128, bias=False), nn.Conv2d(128, 128, 1, bias=False))
print(f"3x3 conv, 128 -> 128: standard {count_params(standard):,} vs depthwise-separable {count_params(separable):,} "
      f"({count_params(standard) / count_params(separable):.1f}x fewer)")

def build_model(pretrained, freeze_backbone):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features                  # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model

m = build_model(pretrained=True, freeze_backbone=False)
print(f"EfficientNet-B0 with {NUM_CLASSES}-class head: {count_params(m):,} parameters")
m_frozen = build_model(pretrained=True, freeze_backbone=True)
print(f"  trainable when the backbone is frozen: {count_params(m_frozen, trainable_only=True):,}")

# Shape trace of the backbone stages for one 224x224 image
m.eval()
x = torch.randn(1, 3, IMG, IMG)
with torch.no_grad():
    for i, layer in enumerate(m.features):
        x = layer(x)
        print(f"features[{i}] -> {tuple(x.shape)}")


# ======================================================================
# 4. Training Setup
# ======================================================================
use_amp = DEVICE.type == "cuda"
to_dev = lambda t: t.to(DEVICE, non_blocking=True, memory_format=torch.channels_last) if t.dim() == 4 else t.to(DEVICE, non_blocking=True)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, frozen=False):
    training = optimizer is not None
    model.train(training)
    if training and frozen:
        model.features.eval()          # keep BatchNorm statistics of the frozen backbone untouched
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = to_dev(x), to_dev(y)
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


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, L, C = [], [], []
    for x, y in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            out = model(to_dev(x))
        conf, pred = out.float().softmax(1).max(1)
        P.append(pred.cpu()); L.append(y); C.append(conf.cpu())
    return torch.cat(P).numpy(), torch.cat(L).numpy(), torch.cat(C).numpy()


def subset_indices(frac):
    if frac >= 1.0:
        return train_idx
    return train_test_split(train_idx, train_size=frac, stratify=targets[train_idx], random_state=SEED)[0]


def train_experiment(name, pretrained, freeze, frac, epochs):
    print(f"\n=== {name} | pretrained={pretrained} frozen={freeze} data={frac:.0%} epochs={epochs} ===")
    torch.manual_seed(SEED)
    idx = subset_indices(frac)
    train_dl = make_loader(Subset(ds_aug, idx), shuffle=True)

    model = build_model(pretrained, freeze).to(DEVICE).to(memory_format=torch.channels_last)
    groups = [{"params": model.classifier.parameters(), "lr": HEAD_LR}]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": BACKBONE_LR if pretrained else HEAD_LR})
    optimizer = optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, scaler, frozen=freeze)
        va_loss, va_acc = run_epoch(model, val_dl, criterion)
        scheduler.step()
        for k, v in zip(hist, [tr_loss, va_loss, tr_acc, va_acc]):
            hist[k].append(v)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {ep:02d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.4f} | {time.time() - t0:.0f}s")
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    preds, labels, _ = predict(model, test_dl)
    test_acc = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"-> best epoch {best_epoch} | val {best_val:.4f} | test {test_acc:.4f} | macro-F1 {macro_f1:.4f} | {elapsed:.0f}s")
    del train_dl
    return dict(model=model, hist=hist, pretrained=pretrained, frozen=freeze, frac=frac, epochs=epochs,
                n_train=len(idx), trainable=count_params(model, trainable_only=True),
                best_epoch=best_epoch, best_val_acc=best_val, test_acc=test_acc, macro_f1=macro_f1, time=elapsed)

results = {}


# Experiment A: pretrained, frozen backbone (100%)
results["A: pretrained, frozen backbone (100%)"] = train_experiment("A: pretrained, frozen backbone (100%)", pretrained=True, freeze=True, frac=1.0, epochs=3)


# Experiment B: pretrained, fine-tune all (100%)
results["B: pretrained, fine-tune all (100%)"] = train_experiment("B: pretrained, fine-tune all (100%)", pretrained=True, freeze=False, frac=1.0, epochs=4)


# Experiment C: from scratch (100%)
results["C: from scratch (100%)"] = train_experiment("C: from scratch (100%)", pretrained=False, freeze=False, frac=1.0, epochs=4)


# Experiment D: pretrained, fine-tune (5% data)
results["D: pretrained, fine-tune (5% data)"] = train_experiment("D: pretrained, fine-tune (5% data)", pretrained=True, freeze=False, frac=0.05, epochs=15)


# Experiment E: from scratch (5% data)
results["E: from scratch (5% data)"] = train_experiment("E: from scratch (5% data)", pretrained=False, freeze=False, frac=0.05, epochs=15)


# ======================================================================
# 5. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Init": "ImageNet" if r["pretrained"] else "random",
        "Trainable params": f"{r['trainable']:,}",
        "Train images": f"{r['n_train']:,}",
        "Epochs": r["epochs"],
        "Best epoch": r["best_epoch"],
        "Val acc (%)": round(r["best_val_acc"] * 100, 2),
        "Test acc (%)": round(r["test_acc"] * 100, 2),
        "Macro-F1 (%)": round(r["macro_f1"] * 100, 2),
        "Time (s)": round(r["time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(summary.to_string(index=False))

# Training curves
names = list(results)
fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7))
for j, name in enumerate(names):
    h = results[name]["hist"]; ep = np.arange(1, len(h["val_acc"]) + 1)
    axes[0, j].plot(ep, np.array(h["train_acc"]) * 100, "--", label="train (augmented)")
    axes[0, j].plot(ep, np.array(h["val_acc"]) * 100, label="val")
    axes[0, j].set_title(name.split("(")[0].strip() + "\n(" + name.split("(")[1], fontsize=9); axes[0, j].set_xlabel("epoch")
    axes[1, j].plot(ep, h["train_loss"], "--", label="train"); axes[1, j].plot(ep, h["val_loss"], label="val")
    axes[1, j].set_xlabel("epoch")
    if j == 0:
        axes[0, j].set_ylabel("accuracy (%)"); axes[1, j].set_ylabel("loss")
        axes[0, j].legend(); axes[1, j].legend()
plt.suptitle("Training curves")
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# Transfer-learning comparison: test accuracy and validation accuracy over epochs
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
labels_ = [n.split(":")[0] for n in names]
accs = [results[n]["test_acc"] * 100 for n in names]
colors = ["tab:blue" if results[n]["pretrained"] else "tab:gray" for n in names]
bars = ax[0].bar(labels_, accs, color=colors)
for b, a_ in zip(bars, accs):
    ax[0].text(b.get_x() + b.get_width() / 2, a_ + 0.3, f"{a_:.2f}", ha="center", fontsize=9)
ax[0].set_ylim(max(0, min(accs) - 8), 101)
ax[0].set_title("Test accuracy (%)   blue = ImageNet init, gray = random init")
ax[0].set_xlabel("A: frozen | B: fine-tune | C: scratch (100% data)   D: fine-tune | E: scratch (5% data)", fontsize=8)
for n in names:
    h = results[n]["hist"]
    ax[1].plot(np.arange(1, len(h["val_acc"]) + 1), np.array(h["val_acc"]) * 100, marker="o", ms=3, label=n.split("(")[0].strip())
ax[1].set_title("Validation accuracy per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("%")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout()
save_fig("transfer_comparison.png")
plt.show()


# ======================================================================
# Error Analysis (best configuration)
# ======================================================================
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best_model = results[best_name]["model"]
print("Best config by validation accuracy:", best_name, f"| test acc {results[best_name]['test_acc']*100:.2f}%")

preds, labels, conf = predict(best_model, test_dl)
rep = classification_report(labels, preds, labels=range(NUM_CLASSES), output_dict=True, zero_division=0)
print(f"accuracy {rep['accuracy']:.4f} | macro precision {rep['macro avg']['precision']:.4f} | "
      f"macro recall {rep['macro avg']['recall']:.4f} | macro F1 {rep['macro avg']['f1-score']:.4f}")

cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
cm_err = cm.copy(); np.fill_diagonal(cm_err, 0)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_err, cmap="Reds", cbar=True, xticklabels=[short(c) for c in classes],
            yticklabels=[short(c) for c in classes])
plt.xticks(fontsize=6); plt.yticks(fontsize=6)
plt.title(f"Confusions only (diagonal removed) - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
save_fig("confusion_errors.png")
plt.show()

top = np.dstack(np.unravel_index(np.argsort(cm_err.ravel())[::-1][:8], cm.shape))[0]
print("Top confusions (true -> predicted : count)")
for t, p in top:
    print(f"  {classes[t]} -> {classes[p]} : {cm_err[t, p]}")

# Per-class F1: the 12 weakest classes
f1s = f1_score(labels, preds, average=None, labels=range(NUM_CLASSES), zero_division=0)
order = np.argsort(f1s)[:12]
plt.figure(figsize=(9, 4.5))
plt.barh([short(classes[i]) for i in order][::-1], (f1s[order] * 100)[::-1], color="tab:orange")
plt.xlabel("F1 (%)"); plt.title(f"Weakest classes - {best_name}")
plt.xlim(max(0, f1s[order].min() * 100 - 5), 100)
plt.tight_layout()
save_fig("weakest_classes.png")
plt.show()
print(f"Mean confidence: correct {conf[preds == labels].mean():.3f} | wrong {conf[preds != labels].mean() if (preds != labels).any() else float('nan'):.3f}")


# ======================================================================
# What Does the Network Look At? Grad-CAM
# ======================================================================
def gradcam(model, x):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    store = {}
    def fwd_hook(mod, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))
    handle = model.features[-1].register_forward_hook(fwd_hook)
    out = model(x.unsqueeze(0).to(DEVICE))
    cls = int(out.argmax(1))
    model.zero_grad(set_to_none=True)
    out[0, cls].backward()
    handle.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), cls

rng = np.random.RandomState(SEED)
correct_ids = rng.choice(np.where(preds == labels)[0], 4, replace=False)
wrong_ids = np.where(preds != labels)[0][:4]
ids = list(correct_ids) + list(wrong_ids)

fig, axes = plt.subplots(2, len(ids), figsize=(2.6 * len(ids), 5.8))
for j, i in enumerate(ids):
    img, lab = test_ds[int(i)]
    cam, cls = gradcam(best_model, img)
    rgb = denorm(img).permute(1, 2, 0).numpy()
    axes[0, j].imshow(rgb); axes[0, j].axis("off")
    axes[0, j].set_title(f"T: {short(classes[lab])}\nP: {short(classes[cls])}", fontsize=7,
                         color="green" if cls == lab else "red")
    axes[1, j].imshow(rgb); axes[1, j].imshow(cam, cmap="jet", alpha=0.45); axes[1, j].axis("off")
plt.suptitle("Grad-CAM (left 4: correct predictions, right 4: errors)")
plt.tight_layout()
save_fig("gradcam.png")
plt.show()

# Misclassified test images
wrong = np.where(preds != labels)[0][:16]
if len(wrong):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), wrong):
        img, lab = test_ds[int(i)]
        a.imshow(denorm(img).permute(1, 2, 0))
        a.set_title(f"T: {short(classes[lab])}\nP: {short(classes[preds[i]])}", fontsize=6.5, color="red")
    plt.suptitle("Misclassified test images (T = true, P = predicted)")
    plt.tight_layout()
    save_fig("misclassified.png")
    plt.show()
else:
    print("No misclassified test images.")

# Save the best model and the class list
torch.save(best_model.state_dict(), OUT / "efficientnet_b0_plant_best.pth")
(OUT / "classes.json").write_text(json.dumps(classes, indent=2))
print("Saved to", OUT)


# ======================================================================
# 6. Analysis
# ======================================================================