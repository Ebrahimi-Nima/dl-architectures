"""
Day 5: U-Net from scratch for brain MRI tumor segmentation (PyTorch).

Script version of unet_brain_mri.ipynb. Runs five experiments (skip connections, loss, augmentation,
and a patient-level vs slice-level split) and saves figures, a results CSV and the best model.
Expects the 'LGG MRI Segmentation' dataset under /kaggle/input (it searches for *_mask.tif files).
A GPU is strongly recommended (about 30-45 minutes on a Kaggle T4). Research and learning only.
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
import os, copy, math, time, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3
BASE = 32            # channels of the first U-Net level (32 -> 64 -> 128 -> 256 -> 512)
THRESH = 0.5         # probability threshold for the predicted mask
MIN_AREA = 10        # a slice counts as "tumor detected" if the predicted mask has >= 10 pixels

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

print("PyTorch:", torch.__version__, "| Device:", DEVICE)


# ======================================================================
# 2. Dataset
# ======================================================================
def list_input(root="/kaggle/input", max_items=12):
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

def describe_input(root="/kaggle/input", n_examples=6):
    """Count the files under /kaggle/input by extension and show a few examples."""
    from collections import Counter
    files = [p for p in Path(root).rglob("*") if p.is_file()] if Path(root).exists() else []
    print(f"{len(files):,} files under {root}")
    by_ext = Counter(p.suffix.lower() for p in files)
    for ext, c in by_ext.most_common(8):
        print(f"  {ext or '(no extension)'}: {c:,}")
    for p in files[:n_examples]:
        print("   e.g.", p)

def find_pairs(root="/kaggle/input"):
    """Return a list of (image_path, mask_path, patient_id), duplicates removed.
    A mask is any image file whose name ends with '_mask' (case-insensitive); its image has the same name without it."""
    import re
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    pairs = {}
    for m in sorted(Path(root).rglob("*")):
        if m.suffix.lower() in exts and re.search(r"_mask$", m.stem, flags=re.I):
            img = m.with_name(re.sub(r"_mask$", "", m.stem, flags=re.I) + m.suffix)
            if img.exists() and m.name not in pairs:
                pairs[m.name] = (img, m, m.parent.name)
    if not pairs:
        list_input(root)
        describe_input(root)
        raise FileNotFoundError("No '*_mask' image files found under /kaggle/input (see the listing above). "
                                "If it is empty, add the 'LGG MRI Segmentation' dataset with '+ Add Input' and re-run.")
    return list(pairs.values())

pairs = find_pairs()
patients_all = sorted({p[2] for p in pairs})
print(f"{len(pairs):,} image/mask pairs from {len(patients_all)} patients")
im0 = Image.open(pairs[0][0]); mk0 = Image.open(pairs[0][1])
print("first image:", im0.mode, im0.size, "| first mask:", mk0.mode, mk0.size)

def load_all(pairs):
    n = len(pairs)
    X = np.zeros((n, 3, 256, 256), np.uint8)
    Y = np.zeros((n, 1, 256, 256), np.uint8)
    t0 = time.time()
    for i, (ip, mp, _) in enumerate(pairs):
        im = Image.open(ip).convert("RGB"); mk = Image.open(mp).convert("L")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BILINEAR); mk = mk.resize((256, 256), Image.NEAREST)
        X[i] = np.asarray(im).transpose(2, 0, 1)
        Y[i, 0] = np.asarray(mk) > 127
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{n} ({time.time() - t0:.0f}s)")
    return X, Y

X_np, Y_np = load_all(pairs)
patient_of = np.array([p[2] for p in pairs])
areas = Y_np.reshape(len(Y_np), -1).sum(1)
has_tumor = areas > 0
print(f"slices with a tumor: {has_tumor.sum():,} ({has_tumor.mean():.1%}) | "
      f"tumor pixels overall: {Y_np.mean():.2%} of all pixels, {Y_np[has_tumor].mean():.2%} within tumor slices")

X = torch.from_numpy(X_np).to(DEVICE)      # uint8 on the GPU
Y = torch.from_numpy(Y_np).to(DEVICE)

# ---- patient-level split (the correct protocol) ----
rng = np.random.RandomState(SEED)
pl = list(patients_all); rng.shuffle(pl)
n_test = int(round(0.15 * len(pl))); n_val = int(round(0.15 * len(pl)))
test_p, val_p, train_p = set(pl[:n_test]), set(pl[n_test:n_test + n_val]), set(pl[n_test + n_val:])
by_patient = tuple(np.where(np.isin(patient_of, list(s)))[0] for s in (train_p, val_p, test_p))

# ---- random slice-level split (leaky; used only in experiment E) ----
perm = np.random.RandomState(SEED).permutation(len(pairs))
n_tr = int(0.70 * len(pairs)); n_va = int(0.15 * len(pairs))
by_slice = (np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + n_va]), np.sort(perm[n_tr + n_va:]))

SPLITS = {"patient": by_patient, "slice": by_slice}

rows = []
for sname, (tr, va, te) in SPLITS.items():
    for part, idx in zip(("train", "val", "test"), (tr, va, te)):
        rows.append({"split": sname, "part": part, "slices": len(idx), "patients": len(set(patient_of[idx])),
                     "tumor slices (%)": round(100 * has_tumor[idx].mean(), 1)})
split_table = pd.DataFrame(rows)
print(split_table.to_string(index=False))

for sname, (tr, va, te) in SPLITS.items():
    shared = set(patient_of[tr]) & set(patient_of[te])
    print(f"{sname:>7} split: {len(shared)} patients appear in BOTH train and test")

# Normalization statistics from the (patient-level) training slices only
tr_idx0 = torch.as_tensor(by_patient[0], device=DEVICE)
s1 = torch.zeros(3, device=DEVICE, dtype=torch.float64); s2 = torch.zeros(3, device=DEVICE, dtype=torch.float64)
cnt = 0
for i in range(0, len(tr_idx0), 256):
    xb = X[tr_idx0[i:i + 256]].double() / 255.0
    s1 += xb.sum((0, 2, 3)); s2 += (xb ** 2).sum((0, 2, 3)); cnt += xb.numel() // 3
MEAN = (s1 / cnt).float().view(1, 3, 1, 1)
STD = ((s2 / cnt - (s1 / cnt) ** 2).sqrt()).float().view(1, 3, 1, 1)
print("channel mean:", MEAN.flatten().tolist(), "| std:", STD.flatten().tolist())

def prep(xb_uint8):
    return (xb_uint8.float() / 255.0 - MEAN) / STD

# Look at the data: samples with ground-truth contours, and the tumor-size distribution
def show(ax, i, mask=None, color="lime"):
    ax.imshow(X_np[i].transpose(1, 2, 0)); ax.axis("off")
    m = Y_np[i, 0] if mask is None else mask
    if m.any():
        ax.contour(m, levels=[0.5], colors=color, linewidths=1.2)

rs = np.random.RandomState(SEED)
pos_ids = rs.choice(np.where(has_tumor)[0], 6, replace=False)
neg_ids = rs.choice(np.where(~has_tumor)[0], 6, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
for ax, i in zip(axes[0], pos_ids):
    show(ax, i); ax.set_title(f"tumor, {areas[i]} px", fontsize=9)
for ax, i in zip(axes[1], neg_ids):
    show(ax, i); ax.set_title("no tumor", fontsize=9)
plt.suptitle("Brain MRI slices (R/G/B = pre-contrast / FLAIR / post-contrast), ground truth in green")
plt.tight_layout()
save_fig("data_samples.png")
plt.show()

plt.figure(figsize=(7, 3.5))
plt.hist(areas[has_tumor], bins=np.logspace(np.log10(max(1, areas[has_tumor].min())), np.log10(areas.max()), 40), color="tab:green")
plt.xscale("log"); plt.xlabel("tumor area in a slice (pixels, log scale)"); plt.ylabel("slices")
plt.title("Tumor size distribution (tumor slices only)")
plt.tight_layout()
save_fig("tumor_size_distribution.png")
plt.show()


# ======================================================================
# 3. Model Implementation
# ======================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, base=32, skip=True):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.skip = skip
        self.enc = nn.ModuleList([DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1]),
                                  DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], c[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[i] * (2 if skip else 1), c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        feats = []
        for blk in self.enc:
            x = blk(x); feats.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, f in zip(self.up, self.dec, reversed(feats)):
            x = up(x)
            if self.skip:
                x = torch.cat([f, x], dim=1)     # skip connection: concatenate encoder features
            x = dec(x)
        return self.head(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())

for sk, expected in [(True, 7_763_041), (False, 6_979_681)]:
    n = count_params(UNet(base=BASE, skip=sk))
    note = f"matches the hand-computed {expected:,}" if (BASE == 32 and n == expected) else "(hand-computed value is for BASE=32)"
    print(f"U-Net base {BASE}, skip={sk}: {n:,} parameters {note}")

x = torch.randn(2, 3, 256, 256)
print("output shape:", tuple(UNet(base=BASE)(x).shape))


# ======================================================================
# 4. Training Setup
# ======================================================================
use_amp = DEVICE.type == "cuda"

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)

def compute_loss(logits, target, kind):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce if kind == "bce" else bce + dice_loss(logits, target)

def augment(x, y):
    """Random affine on a GPU batch: same transform for image (bilinear) and mask (nearest)."""
    B, dev = x.size(0), x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, device=dev)
    ang, sc = u(-15, 15) * math.pi / 180, u(0.9, 1.1)
    tx, ty = u(-0.1, 0.1), u(-0.1, 0.1)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * flip; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * flip; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x, (y > 0.5).float()

@torch.no_grad()
def predict(model, idx, bs=64):
    """Probabilities (N,1,H,W, float16 on device) and mean BCE for the given slice indices."""
    model.eval()
    idx_t = torch.as_tensor(idx, device=DEVICE)
    probs, bce_sum = [], 0.0
    for i in range(0, len(idx_t), bs):
        j = idx_t[i:i + bs]
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(prep(X[j]))
        logits = logits.float()
        bce_sum += F.binary_cross_entropy_with_logits(logits, Y[j].float(), reduction="sum").item()
        probs.append(torch.sigmoid(logits).half())
    probs = torch.cat(probs)
    return probs, bce_sum / (len(idx_t) * 256 * 256)

def slice_stats(probs, idx):
    pred = (probs > THRESH).float()
    g = Y[torch.as_tensor(idx, device=DEVICE)].float()
    inter = (pred * g).sum((1, 2, 3)).double().cpu().numpy()
    p = pred.sum((1, 2, 3)).double().cpu().numpy()
    gs = g.sum((1, 2, 3)).double().cpu().numpy()
    return inter, p, gs

def summarize(inter, p, g):
    pos = g > 0
    dice_slice = 2 * inter / (p + g + 1e-8)
    iou_slice = inter / (p + g - inter + 1e-8)
    return {
        "dice": 2 * inter.sum() / (p.sum() + g.sum() + 1e-8),
        "iou": inter.sum() / (p.sum() + g.sum() - inter.sum() + 1e-8),
        "dice_tumor": dice_slice[pos].mean() if pos.any() else float("nan"),
        "iou_tumor": iou_slice[pos].mean() if pos.any() else float("nan"),
        "sensitivity": (p[pos] >= MIN_AREA).mean() if pos.any() else float("nan"),
        "specificity": (p[~pos] < MIN_AREA).mean() if (~pos).any() else float("nan"),
    }

def train_experiment(name, skip, loss_kind, augment_on, split):
    print(f"\n=== {name} | skip={skip} loss={loss_kind} aug={augment_on} split={split} ===")
    torch.manual_seed(SEED)
    tr_idx, va_idx, te_idx = SPLITS[split]
    tr_t = torch.as_tensor(tr_idx, device=DEVICE)

    model = UNet(base=BASE, skip=skip).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "val_bce": [], "val_dice": []}
    best_dice, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        total, nb = 0.0, 0
        for i in range(0, len(order), BATCH_SIZE):
            j = order[i:i + BATCH_SIZE]
            if len(j) < 2:
                continue
            xb, yb = prep(X[j]), Y[j].float()
            if augment_on:
                xb, yb = augment(xb, yb)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(xb)
            loss = compute_loss(logits.float(), yb, loss_kind)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item(); nb += 1
        scheduler.step()

        probs, vbce = predict(model, va_idx)
        vd = summarize(*slice_stats(probs, va_idx))["dice"]
        hist["train_loss"].append(total / nb); hist["val_bce"].append(vbce); hist["val_dice"].append(vd)
        if vd > best_dice:
            best_dice, best_epoch = vd, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss {total / nb:.4f} | val BCE {vbce:.4f} | val Dice {vd:.4f} | {time.time() - t0:.0f}s")
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    probs, _ = predict(model, te_idx)
    inter, p, g = slice_stats(probs, te_idx)
    metrics = summarize(inter, p, g)
    print(f"-> best epoch {best_epoch} | val Dice {best_dice:.4f} | test Dice {metrics['dice']:.4f} | "
          f"test IoU {metrics['iou']:.4f} | tumor-slice Dice {metrics['dice_tumor']:.4f} | {elapsed:.0f}s")
    return dict(model=model, hist=hist, skip=skip, loss=loss_kind, aug=augment_on, split=split,
                params=count_params(model), best_epoch=best_epoch, best_val_dice=best_dice,
                metrics=metrics, per_slice=(inter, p, g), te_idx=te_idx, time=elapsed)

results = {}


# Experiment A: no skips (BCE+Dice)
results["A: no skips (BCE+Dice)"] = train_experiment("A: no skips (BCE+Dice)", skip=False, loss_kind="bce_dice", augment_on=False, split="patient")


# Experiment B: U-Net (BCE only)
results["B: U-Net (BCE only)"] = train_experiment("B: U-Net (BCE only)", skip=True, loss_kind="bce", augment_on=False, split="patient")


# Experiment C: U-Net (BCE+Dice)
results["C: U-Net (BCE+Dice)"] = train_experiment("C: U-Net (BCE+Dice)", skip=True, loss_kind="bce_dice", augment_on=False, split="patient")


# Experiment D: U-Net + augmentation
results["D: U-Net + augmentation"] = train_experiment("D: U-Net + augmentation", skip=True, loss_kind="bce_dice", augment_on=True, split="patient")


# Experiment E: slice-level split (leaky)
results["E: slice-level split (leaky)"] = train_experiment("E: slice-level split (leaky)", skip=True, loss_kind="bce_dice", augment_on=False, split="slice")


# ======================================================================
# 5. Results
# ======================================================================
rows = []
for name, r in results.items():
    m = r["metrics"]
    rows.append({
        "Config": name,
        "Skips": "yes" if r["skip"] else "no",
        "Loss": r["loss"],
        "Aug": "yes" if r["aug"] else "no",
        "Split": r["split"],
        "Params": f"{r['params']:,}",
        "Best epoch": r["best_epoch"],
        "Val Dice": round(r["best_val_dice"], 4),
        "Test Dice": round(m["dice"], 4),
        "Test IoU": round(m["iou"], 4),
        "Dice (tumor slices)": round(m["dice_tumor"], 4),
        "Sensitivity": round(m["sensitivity"], 3),
        "Specificity": round(m["specificity"], 3),
        "Time (s)": round(r["time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(summary.to_string(index=False))

# Training curves
names = list(results)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
for n in names:
    h = results[n]["hist"]; ep = np.arange(1, len(h["val_dice"]) + 1)
    ax[0].plot(ep, h["val_dice"], label=n)
    ax[1].plot(ep, h["train_loss"], label=n)
    ax[2].plot(ep, h["val_bce"], label=n)
ax[0].set_title("Validation Dice (global)"); ax[0].set_ylim(0.4, 1.0)
ax[1].set_title("Training loss (each run's own loss)")
ax[2].set_title("Validation BCE")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=7)
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# Comparison of the patient-level experiments (A to D)
pat = [n for n in names if results[n]["split"] == "patient"]
metrics_to_plot = [("dice", "Global Dice"), ("dice_tumor", "Dice (tumor slices)"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (k, title) in zip(ax, metrics_to_plot):
    vals = [results[n]["metrics"][k] for n in pat]
    bars = a.bar([n.split(":")[0] for n in pat], vals, color=["tab:gray", "tab:orange", "tab:blue", "tab:green"][:len(pat)])
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    a.set_ylim(max(0, min(vals) - 0.1), min(1.02, max(vals) + 0.06)); a.set_title(title)
plt.suptitle("Experiments A to D (patient-level split, same test patients)")
plt.tight_layout()
save_fig("comparison_AtoD.png")
plt.show()


# ======================================================================
# Qualitative Results and Error Analysis
# ======================================================================
best_name = max(pat, key=lambda k: results[k]["best_val_dice"])
best = results[best_name]
te_idx = best["te_idx"]
inter, p, g = best["per_slice"]
pos = g > 0
dice_slice = 2 * inter / (p + g + 1e-8)
print("Best config by validation Dice:", best_name,
      f"| test Dice {best['metrics']['dice']:.4f} | IoU {best['metrics']['iou']:.4f}")

probs_best, _ = predict(best["model"], te_idx)
pred_masks = (probs_best > THRESH).squeeze(1).cpu().numpy()

rs = np.random.RandomState(SEED)
pos_pos = np.where(pos)[0]
random_ids = rs.choice(pos_pos, 4, replace=False)
worst_ids = pos_pos[np.argsort(dice_slice[pos_pos])[:4]]
ids = list(random_ids) + list(worst_ids)

fig, axes = plt.subplots(3, 8, figsize=(19, 7.2))
for c, k in enumerate(ids):
    gi = te_idx[k]
    axes[0, c].imshow(X_np[gi].transpose(1, 2, 0)); axes[0, c].axis("off")
    axes[0, c].set_title(("random" if c < 4 else "worst") + f" | Dice {dice_slice[k]:.2f}", fontsize=9)
    axes[1, c].imshow(X_np[gi].transpose(1, 2, 0)); axes[1, c].axis("off")
    axes[1, c].contour(Y_np[gi, 0], levels=[0.5], colors="lime", linewidths=1.2)
    axes[2, c].imshow(X_np[gi].transpose(1, 2, 0)); axes[2, c].axis("off")
    if pred_masks[k].any():
        axes[2, c].contour(pred_masks[k], levels=[0.5], colors="red", linewidths=1.2)
axes[0, 0].set_ylabel("image"); 
fig.text(0.005, 0.80, "image", rotation=90, fontsize=11); fig.text(0.005, 0.50, "ground truth", rotation=90, fontsize=11)
fig.text(0.005, 0.17, "prediction", rotation=90, fontsize=11)
plt.suptitle(f"{best_name}: 4 random tumor slices (left) and the 4 worst tumor slices (right)")
plt.tight_layout()
save_fig("predictions_best.png")
plt.show()

# Dice vs tumor size (tumor slices of the best model)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].scatter(g[pos], dice_slice[pos], s=10, alpha=0.5)
ax[0].set_xscale("log"); ax[0].set_xlabel("ground-truth tumor area (pixels, log scale)"); ax[0].set_ylabel("per-slice Dice")
ax[0].set_title("Small tumors are harder"); ax[0].grid(alpha=0.3)
ax[1].hist(dice_slice[pos], bins=25, color="tab:blue")
ax[1].set_xlabel("per-slice Dice (tumor slices)"); ax[1].set_ylabel("slices")
ax[1].set_title(f"Per-slice Dice distribution (median {np.median(dice_slice[pos]):.3f})")
plt.tight_layout()
save_fig("dice_vs_size.png")
plt.show()

small = pos & (g < np.median(g[pos])); large = pos & (g >= np.median(g[pos]))
print(f"mean Dice: smaller half of the tumors {dice_slice[small].mean():.3f} | larger half {dice_slice[large].mean():.3f}")
print(f"tumor slices with Dice < 0.5: {(dice_slice[pos] < 0.5).mean():.1%} | missed completely (no prediction): {(p[pos] < MIN_AREA).mean():.1%}")
print(f"slices without a tumor with a false alarm (>= {MIN_AREA} px): {(p[~pos] >= MIN_AREA).mean():.1%}")

# Effect of the skip connections: same slices, model A (no skips) vs model C (with skips)
rsq = np.random.RandomState(SEED + 1)
a_res, c_res = results["A: no skips (BCE+Dice)"], results["C: U-Net (BCE+Dice)"]
te_idx_ac = c_res["te_idx"]
inter_c, p_c, g_c = c_res["per_slice"]; inter_a, p_a, g_a = a_res["per_slice"]
pos_ac = np.where(g_c > 0)[0]
sel = rsq.choice(pos_ac, 6, replace=False)
pa, _ = predict(a_res["model"], te_idx_ac[sel]); pc, _ = predict(c_res["model"], te_idx_ac[sel])
ma, mc = (pa > THRESH).squeeze(1).cpu().numpy(), (pc > THRESH).squeeze(1).cpu().numpy()

fig, axes = plt.subplots(3, 6, figsize=(16, 8))
for c, k in enumerate(sel):
    gi = te_idx_ac[k]
    for r in range(3):
        axes[r, c].imshow(X_np[gi].transpose(1, 2, 0)); axes[r, c].axis("off")
    axes[0, c].contour(Y_np[gi, 0], levels=[0.5], colors="lime", linewidths=1.2); axes[0, c].set_title("ground truth", fontsize=9)
    if ma[c].any(): axes[1, c].contour(ma[c], levels=[0.5], colors="red", linewidths=1.2)
    axes[1, c].set_title(f"A, no skips | Dice {2 * (ma[c] & Y_np[gi, 0].astype(bool)).sum() / (ma[c].sum() + g_c[k] + 1e-8):.2f}", fontsize=9)
    if mc[c].any(): axes[2, c].contour(mc[c], levels=[0.5], colors="red", linewidths=1.2)
    axes[2, c].set_title(f"C, with skips | Dice {2 * (mc[c] & Y_np[gi, 0].astype(bool)).sum() / (mc[c].sum() + g_c[k] + 1e-8):.2f}", fontsize=9)
plt.suptitle("Skip connections: ground truth (top), no skips (middle), U-Net (bottom)")
plt.tight_layout()
save_fig("skip_comparison.png")
plt.show()


# ======================================================================
# The Leakage Experiment (C vs E)
# ======================================================================
c_m, e_m = results["C: U-Net (BCE+Dice)"]["metrics"], results["E: slice-level split (leaky)"]["metrics"]
keys = [("dice", "Global Dice"), ("iou", "Global IoU"), ("dice_tumor", "Dice (tumor slices)"), ("sensitivity", "Sensitivity")]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(keys)); w = 0.36
b1 = ax.bar(x - w / 2, [c_m[k] for k, _ in keys], w, label="C: split by patient", color="tab:blue")
b2 = ax.bar(x + w / 2, [e_m[k] for k, _ in keys], w, label="E: random slices (leaky)", color="tab:red")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([t for _, t in keys])
lo = min(min(c_m[k], e_m[k]) for k, _ in keys); ax.set_ylim(max(0, lo - 0.1), 1.02)
ax.set_title("Same model and training, different split protocol"); ax.legend()
plt.tight_layout()
save_fig("leakage_comparison.png")
plt.show()

for k, t in keys:
    print(f"{t:<22} patient split {c_m[k]:.4f} | slice split {e_m[k]:.4f} | difference {e_m[k] - c_m[k]:+.4f}")

# Save the best model
torch.save(best["model"].state_dict(), OUT / "unet_best.pth")
print("Saved to", OUT)


# ======================================================================
# 6. Analysis
# ======================================================================