"""
Day 9: Autoencoder, VAE and beta-VAE on Fashion-MNIST (PyTorch).

Script version of autoencoder_vae_fashion_mnist.ipynb. Trains seven models (autoencoders and VAEs with
different latent sizes and beta values), evaluates reconstruction, rate (KL), active units and generation
quality with a Frechet distance in the feature space of a Fashion-MNIST classifier, and saves the figures.
Downloads Fashion-MNIST through torchvision (internet needed). A GPU is recommended (about 15-25 minutes).
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
from scipy import linalg

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
N_SAMPLES = 10000       # generated images per model for the evaluation
ACTIVE_THRESHOLD = 0.01

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("PyTorch:", torch.__version__, "| Device:", DEVICE)


# ======================================================================
# 2. Data
# ======================================================================
def load_fashion_mnist():
    try:
        from torchvision import datasets
        tr = datasets.FashionMNIST("./data", train=True, download=True)
        te = datasets.FashionMNIST("./data", train=False, download=True)
        return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()
    except Exception as e:
        print("torchvision download failed:", repr(e)[:150], "-> trying Hugging Face datasets")
    from datasets import load_dataset
    try:
        ds = load_dataset("zalando-datasets/fashion_mnist")
    except Exception:
        ds = load_dataset("fashion_mnist")
    arr = lambda split: (np.stack([np.array(im) for im in ds[split]["image"]]), np.array(ds[split]["label"]))
    (xtr, ytr), (xte, yte) = arr("train"), arr("test")
    return xtr, ytr, xte, yte

xtr_all, ytr_all, xte_np, yte_np = load_fashion_mnist()
perm = np.random.RandomState(SEED).permutation(len(xtr_all))
va_idx, tr_idx = perm[:5000], perm[5000:]

to_img = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE).div(255.0).unsqueeze(1)
Xtr, Xva, Xte = to_img(xtr_all[tr_idx]), to_img(xtr_all[va_idx]), to_img(xte_np)
Ytr = torch.as_tensor(ytr_all[tr_idx], device=DEVICE); Yva = torch.as_tensor(ytr_all[va_idx], device=DEVICE)
Yte = torch.as_tensor(yte_np, device=DEVICE)
print("train", tuple(Xtr.shape), "| val", tuple(Xva.shape), "| test", tuple(Xte.shape), "| pixel range", Xtr.min().item(), Xtr.max().item())

def show_grid(ax, imgs, ncols):
    """imgs: (N, 1, 28, 28) tensor -> one tiled picture"""
    imgs = imgs.detach().cpu().numpy()[:, 0]
    n = len(imgs); nrows = math.ceil(n / ncols)
    canvas = np.ones((nrows * 28, ncols * 28))
    for k, im in enumerate(imgs):
        r, c = divmod(k, ncols); canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = im
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1); ax.axis("off")

fig, ax = plt.subplots(figsize=(12, 2.6))
show_grid(ax, Xte[:32], 16)
plt.title("Fashion-MNIST test images")
save_fig("data_samples.png")
plt.show()


# ======================================================================
# 3. A Classifier for Evaluation (FD-CNN)
# ======================================================================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 5 * 5, 128), nn.ReLU())
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 10)
    def features(self, x):
        return self.fc(self.conv(x))
    def forward(self, x):
        return self.out(self.drop(self.features(x)))

torch.manual_seed(SEED)
clf = Classifier().to(DEVICE)
opt = optim.Adam(clf.parameters(), lr=1e-3)
for ep in range(6):
    clf.train()
    order = torch.randperm(len(Xtr), device=DEVICE)
    for i in range(0, len(order), 128):
        j = order[i:i + 128]
        loss = F.cross_entropy(clf(Xtr[j]), Ytr[j])
        opt.zero_grad(); loss.backward(); opt.step()
clf.eval()

@torch.no_grad()
def clf_features_logits(X, bs=1000):
    fs, ls = [], []
    for i in range(0, len(X), bs):
        f = clf.features(X[i:i + bs]); fs.append(f); ls.append(clf.out(f))
    return torch.cat(fs).cpu().numpy().astype(np.float64), torch.cat(ls)

feat_te, logits_te = clf_features_logits(Xte)
clf_acc = (logits_te.argmax(1) == Yte).float().mean().item()
print(f"classifier test accuracy: {clf_acc:.4f}")

def stats(f):
    return f.mean(0), np.cov(f, rowvar=False)

def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = np.real(covmean)
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))

mu_te, cov_te = stats(feat_te)
feat_tr_ref, _ = clf_features_logits(Xtr[:10000])
FD_FLOOR = frechet(*stats(feat_tr_ref), mu_te, cov_te)
conf_real = torch.softmax(logits_te, 1).max(1).values.mean().item()
print(f"FD-CNN between 10,000 real training images and the real test images (floor): {FD_FLOOR:.3f}")
print(f"mean classifier confidence on real test images: {conf_real:.3f}")


# ======================================================================
# 4. Models
# ======================================================================
class Encoder(nn.Module):
    def __init__(self, z_dim, variational):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(64 * 7 * 7, z_dim)
        self.logvar = nn.Linear(64 * 7 * 7, z_dim) if variational else None
    def forward(self, x):
        h = self.net(x)
        return self.mu(h), (self.logvar(h).clamp(-10, 10) if self.logvar is not None else None)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, 64 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 7, 7))          # logits; sigmoid gives pixel probabilities

class AutoEncoder(nn.Module):
    variational = False
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim, self.enc, self.dec = z_dim, Encoder(z_dim, False), Decoder(z_dim)
    def code(self, x):                                          # deterministic code
        return self.enc(x)[0]
    def forward(self, x):
        z = self.code(x)
        return self.dec(z), z

class VAE(nn.Module):
    variational = True
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim, self.enc, self.dec = z_dim, Encoder(z_dim, True), Decoder(z_dim)
    def code(self, x):                                          # posterior mean
        return self.enc(x)[0]
    def forward(self, x):
        mu, logvar = self.enc(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterization trick
        return self.dec(z), mu, logvar

def kl_per_image(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)

def rec_per_image(logits, x):
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum((1, 2, 3))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

for name, m in [("AutoEncoder z=16", AutoEncoder(16)), ("VAE z=16", VAE(16)), ("AutoEncoder z=2", AutoEncoder(2)), ("VAE z=2", VAE(2))]:
    print(f"{name:18s} {count_params(m):,} parameters")
x = Xtr[:4].cpu()
print("VAE forward shapes:", [tuple(t.shape) for t in VAE(16)(x)], "| AE forward shapes:", [tuple(t.shape) for t in AutoEncoder(16)(x)])


# ======================================================================
# 5. Training and Evaluation
# ======================================================================
def run_epoch(model, X, beta, optimizer=None):
    train = optimizer is not None
    model.train(train)
    order = torch.randperm(len(X), device=DEVICE) if train else torch.arange(len(X), device=DEVICE)
    tot_rec, tot_kl, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(order), BATCH_SIZE):
            x = X[order[i:i + BATCH_SIZE]]
            if model.variational:
                logits, mu, logvar = model(x)
                rec, kl = rec_per_image(logits, x).mean(), kl_per_image(mu, logvar).mean()
                loss = rec + beta * kl
            else:
                logits, _ = model(x)
                rec = rec_per_image(logits, x).mean(); kl = torch.zeros((), device=DEVICE); loss = rec
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            tot_rec += rec.item() * len(x); tot_kl += kl.item() * len(x); n += len(x)
    return tot_rec / n, tot_kl / n

@torch.no_grad()
def codes_of(model, X, bs=1000):
    model.eval()
    return torch.cat([model.code(X[i:i + bs]) for i in range(0, len(X), bs)])

@torch.no_grad()
def decode_np_z(model, z_np, bs=1000):
    model.eval()
    out = []
    for i in range(0, len(z_np), bs):
        z = torch.as_tensor(z_np[i:i + bs], dtype=torch.float32, device=DEVICE)
        out.append(torch.sigmoid(model.dec(z)))
    return torch.cat(out)

def generation_metrics(model, z_np):
    imgs = decode_np_z(model, z_np)
    f, logits = clf_features_logits(imgs)
    p = torch.softmax(logits, 1)
    hist = torch.bincount(p.argmax(1), minlength=10).float(); hist = hist / hist.sum()
    entropy = float(-(hist[hist > 0] * hist[hist > 0].log()).sum())
    return dict(fd=frechet(*stats(f), mu_te, cov_te), conf=p.max(1).values.mean().item(), entropy=entropy), imgs[:64].cpu()

@torch.no_grad()
def evaluate(model, beta):
    model.eval()
    rec_sum, kl_sum, mse_sum, n = 0.0, 0.0, 0.0, 0
    mus, kls, recon_imgs = [], [], []
    for i in range(0, len(Xte), 1000):
        x = Xte[i:i + 1000]
        if model.variational:
            logits, mu, logvar = model(x)
            kl_sum += kl_per_image(mu, logvar).sum().item(); mus.append(mu)
            kls.append(0.5 * (mu ** 2 + logvar.exp() - logvar - 1))
            recon = torch.sigmoid(model.dec(mu))
        else:
            logits, z = model(x); recon = torch.sigmoid(logits)
        rec_sum += rec_per_image(logits, x).sum().item(); n += len(x)
        mse_sum += ((recon - x) ** 2).mean((1, 2, 3)).sum().item(); recon_imgs.append(recon)
    recon_imgs = torch.cat(recon_imgs)
    r = dict(rec_bce=rec_sum / n, mse=mse_sum / n)
    r["recon_acc"] = (clf_features_logits(recon_imgs)[1].argmax(1) == Yte).float().mean().item()
    if model.variational:
        mu_all = torch.cat(mus)
        r["kl"] = kl_sum / n; r["neg_elbo"] = r["rec_bce"] + r["kl"]
        r["active"] = int((mu_all.var(0) > ACTIVE_THRESHOLD).sum().item())
        r["kl_per_dim"] = torch.cat(kls).mean(0).cpu().numpy()
    # generation: prior N(0, I) and a Gaussian fitted to the training codes
    rng = np.random.RandomState(SEED)
    g_prior, imgs_prior = generation_metrics(model, rng.randn(N_SAMPLES, model.z_dim))
    ctr = codes_of(model, Xtr).cpu().numpy().astype(np.float64)
    cov = np.atleast_2d(np.cov(ctr, rowvar=False)) + 1e-6 * np.eye(model.z_dim)
    g_fit, imgs_fit = generation_metrics(model, rng.multivariate_normal(ctr.mean(0), cov, N_SAMPLES))
    r.update(fd_prior=g_prior["fd"], conf_prior=g_prior["conf"], ent_prior=g_prior["entropy"],
             fd_fit=g_fit["fd"], conf_fit=g_fit["conf"], ent_fit=g_fit["entropy"])
    r["samples_prior"], r["samples_fit"] = imgs_prior, imgs_fit
    return r

def train_experiment(name, kind, z_dim, beta):
    print(f"\n=== {name} | {kind} | z={z_dim} | beta={beta} ===")
    torch.manual_seed(SEED)
    model = (VAE(z_dim) if kind == "vae" else AutoEncoder(z_dim)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    hist = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr_rec, tr_kl = run_epoch(model, Xtr, beta, optimizer)
        va_rec, va_kl = run_epoch(model, Xva, beta)
        for k, v in zip(hist, [tr_rec, tr_kl, va_rec, va_kl]):
            hist[k].append(v)
        if ep == 1 or ep % 5 == 0:
            print(f"Epoch {ep:02d}/{EPOCHS} | train rec {tr_rec:.2f} KL {tr_kl:.2f} | val rec {va_rec:.2f} KL {va_kl:.2f} | {time.time() - t0:.0f}s")
    elapsed = time.time() - t0
    r = evaluate(model, beta)
    print(f"-> test: rec BCE {r['rec_bce']:.2f} | MSE {r['mse']:.4f}" +
          (f" | KL {r['kl']:.2f} | active units {r['active']}/{z_dim}" if kind == "vae" else "") +
          f" | FD-CNN prior {r['fd_prior']:.2f}, fitted {r['fd_fit']:.2f} | recon acc {r['recon_acc']:.3f}")
    r.update(model=model, hist=hist, kind=kind, z_dim=z_dim, beta=beta, params=count_params(model), time=elapsed)
    return r

results = {}


# Experiment A: AE (z=16)
results["A: AE (z=16)"] = train_experiment("A: AE (z=16)", "ae", 16, 0.0)


# Experiment B: VAE (z=16, beta=1)
results["B: VAE (z=16, beta=1)"] = train_experiment("B: VAE (z=16, beta=1)", "vae", 16, 1.0)


# Experiment C: VAE (z=16, beta=0.1)
results["C: VAE (z=16, beta=0.1)"] = train_experiment("C: VAE (z=16, beta=0.1)", "vae", 16, 0.1)


# Experiment D: VAE (z=16, beta=4)
results["D: VAE (z=16, beta=4)"] = train_experiment("D: VAE (z=16, beta=4)", "vae", 16, 4.0)


# Experiment E: VAE (z=16, beta=10)
results["E: VAE (z=16, beta=10)"] = train_experiment("E: VAE (z=16, beta=10)", "vae", 16, 10.0)


# Experiment F: AE (z=2)
results["F: AE (z=2)"] = train_experiment("F: AE (z=2)", "ae", 2, 0.0)


# Experiment G: VAE (z=2, beta=1)
results["G: VAE (z=2, beta=1)"] = train_experiment("G: VAE (z=2, beta=1)", "vae", 2, 1.0)


# ======================================================================
# 6. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "z": r["z_dim"],
        "Params": f"{r['params']:,}",
        "Rec BCE (nats/img)": round(r["rec_bce"], 2),
        "Rec MSE (per pixel)": round(r["mse"], 4),
        "KL (nats/img)": round(r["kl"], 2) if r["kind"] == "vae" else None,
        "-ELBO (beta=1)": round(r["neg_elbo"], 2) if r["kind"] == "vae" else None,
        "Active units": f"{r['active']}/{r['z_dim']}" if r["kind"] == "vae" else None,
        "Recon accuracy (%)": round(r["recon_acc"] * 100, 1),
        "FD-CNN (N(0,I))": round(r["fd_prior"], 2),
        "FD-CNN (fitted Gaussian)": round(r["fd_fit"], 2),
        "Confidence (N(0,I))": round(r["conf_prior"], 3),
        "Class entropy (N(0,I))": round(r["ent_prior"], 2),
        "Time (s)": round(r["time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(f"References: FD-CNN floor between two sets of real images = {FD_FLOOR:.2f} | classifier accuracy on real test images = "
      f"{clf_acc * 100:.1f}% | mean confidence on real images = {conf_real:.3f} | maximum class entropy = {math.log(10):.2f}")
print(summary.to_string(index=False))

# Training curves (validation set): reconstruction and KL per image
names = list(results)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
for n in names:
    h = results[n]["hist"]; ep = np.arange(1, EPOCHS + 1)
    ax[0].plot(ep, h["val_rec"], label=n)
    if results[n]["kind"] == "vae":
        ax[1].plot(ep, h["val_kl"], label=n)
ax[0].set_title("Validation reconstruction loss (BCE per image)"); ax[1].set_title("Validation KL per image (VAEs)")
ax[0].set_ylim(top=min(max(results[n]["hist"]["val_rec"][0] for n in names), 400))
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

vae16 = [n for n in names if results[n]["kind"] == "vae" and results[n]["z_dim"] == 16]
vae16.sort(key=lambda n: results[n]["beta"])
betas = [results[n]["beta"] for n in vae16]
ae16 = next(n for n in names if results[n]["kind"] == "ae" and results[n]["z_dim"] == 16)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
ax[0].plot(betas, [results[n]["kl"] for n in vae16], marker="o"); ax[0].set_xscale("log")
ax[0].set_xlabel("beta"); ax[0].set_ylabel("rate: KL (nats per image)"); ax[0].set_title("Rate vs beta")

kls, mses = [results[n]["kl"] for n in vae16], [results[n]["mse"] for n in vae16]
ax[1].plot(kls, mses, marker="o")
for b, k, m in zip(betas, kls, mses):
    ax[1].annotate(f"beta={b}", (k, m), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax[1].axhline(results[ae16]["mse"], color="gray", ls="--", label="autoencoder (z=16)")
ax[1].set_xlabel("rate: KL (nats per image)"); ax[1].set_ylabel("distortion: reconstruction MSE per pixel")
ax[1].set_title("Rate-distortion curve"); ax[1].legend()

ax[2].plot(betas, [results[n]["fd_prior"] for n in vae16], marker="o", label="VAE, z ~ N(0, I)")
ax[2].plot(betas, [results[n]["fd_fit"] for n in vae16], marker="s", label="VAE, fitted Gaussian")
ax[2].axhline(results[ae16]["fd_prior"], color="tab:red", ls="--", label="AE, z ~ N(0, I)")
ax[2].axhline(results[ae16]["fd_fit"], color="tab:green", ls="--", label="AE, fitted Gaussian")
ax[2].axhline(FD_FLOOR, color="k", ls=":", label="real vs real (floor)")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("beta"); ax[2].set_ylabel("FD-CNN (lower is better)")
ax[2].set_title("Generation quality vs beta"); ax[2].legend(fontsize=7)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
save_fig("rate_distortion.png")
plt.show()

with torch.no_grad():
    originals = Xte[:16]
    recon_rows = [("original", originals)]
    for n in [ae16] + vae16:
        m = results[n]["model"].eval()
        r_ = torch.sigmoid(m(originals)[0]) if not m.variational else torch.sigmoid(m.dec(m.code(originals)))
        recon_rows.append((n, r_))

fig, axes = plt.subplots(len(recon_rows), 1, figsize=(12, 1.15 * len(recon_rows)))
for ax, (label, imgs) in zip(axes, recon_rows):
    show_grid(ax, imgs, 16); ax.set_title(label, fontsize=8, loc="left")
plt.tight_layout()
save_fig("reconstructions.png")
plt.show()

sample_rows = [(ae16 + "  | z ~ N(0, I)", results[ae16]["samples_prior"][:16]),
               (ae16 + "  | fitted Gaussian", results[ae16]["samples_fit"][:16])]
for n in vae16:
    sample_rows.append((n + "  | z ~ N(0, I)", results[n]["samples_prior"][:16]))
fig, axes = plt.subplots(len(sample_rows), 1, figsize=(12, 1.15 * len(sample_rows)))
for ax, (label, imgs) in zip(axes, sample_rows):
    show_grid(ax, imgs, 16); ax.set_title(label, fontsize=8, loc="left")
plt.tight_layout()
save_fig("samples.png")
plt.show()

f_name, g_name = next(n for n in names if n.startswith("F:")), next(n for n in names if n.startswith("G:"))
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, n in zip(ax, [f_name, g_name]):
    z = codes_of(results[n]["model"], Xte[:5000]).cpu().numpy()
    sc = a.scatter(z[:, 0], z[:, 1], c=yte_np[:5000], cmap="tab10", s=5, alpha=0.7)
    a.set_title(f"{n}: codes of 5,000 test images"); a.set_xlabel("z1"); a.set_ylabel("z2"); a.grid(alpha=0.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 9), label=c) for i, c in enumerate(CLASSES)]
ax[1].legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
plt.tight_layout()
save_fig("latent_2d.png")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
for a, n in zip(ax, [f_name, g_name]):
    zc = codes_of(results[n]["model"], Xtr).cpu().numpy()
    if results[n]["kind"] == "vae":
        lo, hi = np.array([-3.0, -3.0]), np.array([3.0, 3.0])
    else:
        lo, hi = np.percentile(zc, 1, axis=0), np.percentile(zc, 99, axis=0)
    g1, g2 = np.linspace(lo[0], hi[0], 15), np.linspace(hi[1], lo[1], 15)
    grid = np.array([[x1, x2] for x2 in g2 for x1 in g1])
    imgs = decode_np_z(results[n]["model"], grid)
    show_grid(a, imgs, 15)
    a.set_title(f"{n}: decoded grid over z1 in [{lo[0]:.1f}, {hi[0]:.1f}], z2 in [{lo[1]:.1f}, {hi[1]:.1f}]", fontsize=9)
plt.tight_layout()
save_fig("manifold_2d.png")
plt.show()

first_of = lambda c: int((Yte == c).nonzero()[0])
pairs = [(first_of(0), first_of(7)), (first_of(3), first_of(9)), (first_of(8), first_of(1))]   # T-shirt->Sneaker, Dress->Ankle boot, Bag->Trouser
models_i = [ae16, "B: VAE (z=16, beta=1)"]
fig, axes = plt.subplots(len(pairs) * 2, 1, figsize=(10, 1.3 * len(pairs) * 2))
row = 0
with torch.no_grad():
    for (a, b) in pairs:
        for n in models_i:
            m = results[n]["model"].eval()
            za, zb = m.code(Xte[a:a + 1]), m.code(Xte[b:b + 1])
            ts = torch.linspace(0, 1, 9, device=DEVICE).view(-1, 1)
            imgs = torch.sigmoid(m.dec((1 - ts) * za + ts * zb))
            show_grid(axes[row], imgs, 9); axes[row].set_title(n, fontsize=8, loc="left"); row += 1
plt.tight_layout()
save_fig("interpolation.png")
plt.show()

vae_show = [n for n in vae16 if results[n]["beta"] in (1.0, 4.0, 10.0)]
fig, ax = plt.subplots(1, len(vae_show), figsize=(5 * len(vae_show), 3.6), sharey=True)
for a, n in zip(np.atleast_1d(ax), vae_show):
    kd = results[n]["kl_per_dim"]
    a.bar(range(len(kd)), kd, color=["tab:blue" if v > 0.05 else "lightgray" for v in kd])
    a.set_title(f"{n}\nactive units: {results[n]['active']}/16", fontsize=9); a.set_xlabel("latent dimension")
np.atleast_1d(ax)[0].set_ylabel("average KL (nats)")
plt.tight_layout()
save_fig("per_dim_kl.png")
plt.show()

trav = [n for n in vae16 if results[n]["beta"] in (1.0, 4.0)]
vals = torch.linspace(-3, 3, 9, device=DEVICE)
fig, axes = plt.subplots(5, len(trav), figsize=(6.4 * len(trav), 7.2))
axes = axes.reshape(5, len(trav))
with torch.no_grad():
    for col, n in enumerate(trav):
        top = np.argsort(-results[n]["kl_per_dim"])[:5]
        m = results[n]["model"].eval()
        for row, d in enumerate(top):
            z = torch.zeros(9, 16, device=DEVICE); z[:, d] = vals
            show_grid(axes[row, col], torch.sigmoid(m.dec(z)), 9)
            axes[row, col].set_title(f"{n}: dimension {d} (KL {results[n]['kl_per_dim'][d]:.2f}), from -3 to +3", fontsize=8, loc="left")
plt.tight_layout()
save_fig("traversals.png")
plt.show()

# Save the best VAE (lowest -ELBO among the beta = 1 models) and the evaluation classifier
b_name = "B: VAE (z=16, beta=1)"
torch.save(results[b_name]["model"].state_dict(), OUT / "vae_z16.pth")
torch.save(clf.state_dict(), OUT / "evaluation_classifier.pth")
print("Saved to", OUT)


# ======================================================================
# 7. Analysis
# ======================================================================