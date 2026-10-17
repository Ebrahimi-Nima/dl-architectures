"""
Day 8: Transformer encoder from scratch and fine-tuned BERT/DistilBERT for IMDB sentiment (PyTorch).

Script version of transformer_bert_imdb.ipynb. Uses the same IMDB split as Day 6. Trains a from-scratch
Transformer encoder (with and without positional encodings) and fine-tunes DistilBERT / BERT-base with
Hugging Face transformers. Needs internet (datasets + pretrained weights) and a GPU (about 45-60 minutes).
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
import os, re, copy, math, time, json, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

SEED = 42
RUN_BERT_BASE = True          # set to False to skip experiment F (saves about 10-15 minutes)

# --- from-scratch Transformer (experiments A and B) ---
VOCAB_SIZE, MAX_LEN = 20000, 400          # same as Day 6
D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT = 128, 4, 2, 512, 0.1
S_EPOCHS, S_BATCH, S_LR, S_WARMUP = 10, 64, 5e-4, 300

# --- pretrained models (experiments C to F) ---
P_BATCH = 32
LR_FINETUNE, LR_HEAD = 2e-5, 1e-3

DAY6_GRU_TEST_ACC = 87.84     # best Day 6 model (GRU), same split

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = DEVICE.type == "cuda"

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")

print("PyTorch:", torch.__version__, "| Device:", DEVICE)


# ======================================================================
# 2. Dataset
# ======================================================================
def load_imdb():
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("stanfordnlp/imdb")
        except Exception:
            ds = load_dataset("imdb")
        return ("official", list(ds["train"]["text"]), list(ds["train"]["label"]),
                list(ds["test"]["text"]), list(ds["test"]["label"]))
    except Exception as e:
        print("Could not load the dataset from Hugging Face:", repr(e)[:150])
    for p in sorted(Path("/kaggle/input").rglob("*.csv")):
        try:
            head = pd.read_csv(p, nrows=2)
        except Exception:
            continue
        if {"review", "sentiment"} <= set(head.columns):
            df = pd.read_csv(p).drop_duplicates(subset="review")
            y = (df["sentiment"].str.lower() == "positive").astype(int).values
            tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
            texts = df["review"].tolist()
            print(f"WARNING: using {p} with a RANDOM split; scores are optimistic and NOT comparable with Day 6.")
            return "csv", [texts[i] for i in tr], y[tr].tolist(), [texts[i] for i in te], y[te].tolist()
    raise RuntimeError("Dataset not found. Turn Internet ON (Settings), or add 'IMDB Dataset of 50K Movie Reviews' via '+ Add Input'.")

source, tr_texts_all, tr_labels_all, te_texts, te_labels = load_imdb()
print("data source:", source)

tr_idx, va_idx = train_test_split(np.arange(len(tr_texts_all)), test_size=0.10, stratify=tr_labels_all, random_state=SEED)
train_texts = [tr_texts_all[i] for i in tr_idx]; train_labels = [tr_labels_all[i] for i in tr_idx]
val_texts   = [tr_texts_all[i] for i in va_idx]; val_labels   = [tr_labels_all[i] for i in va_idx]

train_set = set(train_texts)
keep_v = [i for i, t in enumerate(val_texts) if t not in train_set]
keep_t = [i for i, t in enumerate(te_texts) if t not in train_set]
print(f"duplicates of training reviews removed: {len(val_texts) - len(keep_v)} from validation, {len(te_texts) - len(keep_t)} from test")
val_texts = [val_texts[i] for i in keep_v]; val_labels = [val_labels[i] for i in keep_v]
te_texts  = [te_texts[i] for i in keep_t];  te_labels  = [te_labels[i] for i in keep_t]
ytr_np, yva_np, yte_np = np.array(train_labels), np.array(val_labels), np.array(te_labels)
print(f"sizes: train {len(train_texts):,} | val {len(val_texts):,} | test {len(te_texts):,}"
      + ("  (same test set as Day 6)" if len(te_texts) == 24893 else "  (Day 6 had 24,893 test reviews)"))

# Word-level tokenization and vocabulary for the from-scratch Transformer (identical to Day 6)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("<br />", " "))

counter = Counter(tok for t in train_texts for tok in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common(VOCAB_SIZE) if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}

def encode(texts):
    n = len(texts)
    X = np.zeros((n, MAX_LEN), np.int64); L = np.zeros(n, np.int64); O = np.zeros(n, np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tokenize(t)] or [1]
        O[i] = len(ids); ids = ids[:MAX_LEN]; L[i] = len(ids); X[i, :L[i]] = ids
    return X, L, O

Xtr_np, Ltr_np, Otr_np = encode(train_texts)
Xva_np, Lva_np, Ova_np = encode(val_texts)
Xte_np, Lte_np, Ote_np = encode(te_texts)      # Ote_np = original review length in tokens (before truncation)
print("vocabulary size:", len(itos))

to_t = lambda a, dt=None: torch.as_tensor(a, device=DEVICE, dtype=dt)
Xtr, Ltr = to_t(Xtr_np), to_t(Ltr_np); Xva, Lva = to_t(Xva_np), to_t(Lva_np); Xte, Lte = to_t(Xte_np), to_t(Lte_np)
ytr, yva = to_t(ytr_np, torch.float32), to_t(yva_np, torch.float32)


# ======================================================================
# 3. A Transformer Encoder from Scratch
# ======================================================================
def sinusoidal(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1).float()
    freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq); pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, real):                       # x: (B, L, D); real: (B, L) True for real tokens
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)     # each (B, h, L, dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                           # (B, h, L, L)
        scores = scores.masked_fill(~real[:, None, None, :], float("-inf"))              # never attend to padding
        attn = torch.softmax(scores, dim=-1)
        ctx = (self.drop(attn) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(ctx), attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, real):
        a, attn = self.attn(self.ln1(x), real)
        x = x + self.drop(a)                           # residual connection
        x = x + self.drop(self.ff(self.ln2(x)))        # residual connection
        return x, attn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, use_pe=True, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                 dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.d_model, self.use_pe = d_model, use_pe
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model ** -0.5)
        self.emb.weight.data[0].zero_()
        self.register_buffer("pe", sinusoidal(max_len, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        real = x != 0
        h = self.emb(x) * math.sqrt(self.d_model)
        if self.use_pe:
            h = h + self.pe[:x.size(1)]
        h = self.drop(h)
        attns = []
        for layer in self.layers:
            h, a = layer(h, real)
            if return_attn:
                attns.append(a)
        h = self.norm(h)
        m = real.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)                 # masked mean pooling
        logit = self.fc(self.drop(pooled)).squeeze(1)
        return (logit, attns) if return_attn else logit


def count_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))

m = TransformerClassifier(len(itos))
print(f"my Transformer encoder: {count_params(m):,} parameters (embedding table: {m.emb.weight.numel():,})")

# Sanity check: without positional encoding the model is permutation-invariant over the tokens
m.eval()
x = Xtr[:1, :int(Ltr[0])].cpu()
perm = x[:, torch.randperm(x.size(1))]
m_nope = TransformerClassifier(len(itos), use_pe=False).eval()
with torch.no_grad():
    d_nope = (m_nope(x) - m_nope(perm)).abs().item()
    d_pe = (m(x) - m(perm)).abs().item()
print(f"output change when the words are shuffled: without positional encoding {d_nope:.2e} (should be ~0), with it {d_pe:.2e}")


# ======================================================================
# 4. Training Setup
# ======================================================================
def ece_score(conf, correct, n_bins=10):
    """Expected calibration error with equal-width confidence bins."""
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e

def finalize(name, family, model, hist, best_epoch, best_val, probs, train_time, infer_time, trainable):
    pred = (probs > 0.5).astype(int)
    acc = float((pred == yte_np).mean()); f1 = f1_score(yte_np, pred)
    print(f"-> best epoch {best_epoch} | val acc {best_val:.4f} | test acc {acc:.4f} | F1 {f1:.4f} | "
          f"train {train_time:.0f}s | inference {infer_time:.0f}s")
    return dict(model=model, family=family, hist=hist, params=count_params(model), trainable=trainable,
                best_epoch=best_epoch, best_val_acc=best_val, test_acc=acc, test_f1=f1, test_probs=probs,
                time=train_time, infer_time=infer_time)

# ---------------- from-scratch Transformer ----------------
@torch.no_grad()
def predict_scratch(model, X, L, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs][:, :int(L[i:i + bs].max())]
        out.append(torch.sigmoid(model(x)).cpu())
    return torch.cat(out).numpy()

def train_scratch(name, use_pe):
    print(f"\n=== {name} | positional encoding={use_pe} ===")
    torch.manual_seed(SEED)
    model = TransformerClassifier(len(itos), use_pe=use_pe).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=S_LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(Xtr) / S_BATCH); total = steps_per_epoch * S_EPOCHS
    lr_at = lambda s: min((s + 1) / S_WARMUP, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    hist = {"train_loss": [], "val_acc": []}
    best_val, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, S_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xtr), device=DEVICE)
        tot = 0.0
        for i in range(0, len(order), S_BATCH):
            j = order[i:i + S_BATCH]
            x = Xtr[j][:, :int(Ltr[j].max())]
            loss = F.binary_cross_entropy_with_logits(model(x), ytr[j])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step()
            tot += loss.item() * len(j)
        va = float(((predict_scratch(model, Xva, Lva) > 0.5) == yva_np).mean())
        hist["train_loss"].append(tot / len(Xtr)); hist["val_acc"].append(va)
        if va > best_val:
            best_val, best_epoch = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {ep}/{S_EPOCHS} | train loss {tot / len(Xtr):.3f} | val acc {va:.4f} | {time.time() - t0:.0f}s")
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    t1 = time.time(); probs = predict_scratch(model, Xte, Lte); infer_time = time.time() - t1
    return finalize(name, "scratch", model, hist, best_epoch, best_val, probs, train_time, infer_time, count_params(model, True))

results = {}


# Experiment A: my Transformer (+ positions)
results["A: my Transformer (+ positions)"] = train_scratch("A: my Transformer (+ positions)", True)


# Experiment B: my Transformer (no positions)
results["B: my Transformer (no positions)"] = train_scratch("B: my Transformer (no positions)", False)


# ======================================================================
# 5. Pretrained Models
# ======================================================================
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

_tok_cache = {}
def get_tokens(max_len):
    if max_len not in _tok_cache:
        tk = AutoTokenizer.from_pretrained("bert-base-uncased")
        def enc(texts):
            e = tk([t.replace("<br />", " ") for t in texts], truncation=True, max_length=max_len,
                   padding="max_length", return_tensors="np")
            return to_t(e["input_ids"]), to_t(e["attention_mask"])
        t0 = time.time()
        _tok_cache[max_len] = {"train": enc(train_texts), "val": enc(val_texts), "test": enc(te_texts)}
        print(f"tokenized all reviews with max length {max_len} in {time.time() - t0:.0f}s")
    return _tok_cache[max_len]

@torch.no_grad()
def predict_hf(model, ids, mask, bs=128):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        m = mask[i:i + bs]; L = int(m.sum(1).max())
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(input_ids=ids[i:i + bs, :L], attention_mask=m[:, :L]).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(out).numpy()

def train_hf(name, model_name, max_len, epochs, freeze):
    print(f"\n=== {name} | {model_name} | max length {max_len} | epochs {epochs} | frozen backbone={freeze} ===")
    torch.manual_seed(SEED)
    tokens = get_tokens(max_len)
    (Xi, Xm), (Vi, Vm), (Ti, Tm) = tokens["train"], tokens["val"], tokens["test"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
    if freeze:
        for p in model.base_model.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"parameters: {count_params(model):,} total | {count_params(model, True):,} trainable")
    optimizer = optim.AdamW(params, lr=LR_HEAD if freeze else LR_FINETUNE, weight_decay=0.01)
    total = math.ceil(len(Xi) / P_BATCH) * epochs
    sched = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "val_acc": []}
    best_val, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if freeze:
            model.base_model.eval()                    # frozen backbone: no dropout
        order = torch.randperm(len(Xi), device=DEVICE)
        tot = 0.0
        for i in range(0, len(order), P_BATCH):
            j = order[i:i + P_BATCH]
            m = Xm[j]; L = int(m.sum(1).max())
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(input_ids=Xi[j][:, :L], attention_mask=m[:, :L]).logits
            loss = F.cross_entropy(logits.float(), ytr[j].long())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer); scaler.update(); sched.step()
            tot += loss.item() * len(j)
        va = float(((predict_hf(model, Vi, Vm) > 0.5) == yva_np).mean())
        hist["train_loss"].append(tot / len(Xi)); hist["val_acc"].append(va)
        if va > best_val:
            best_val, best_epoch = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {ep}/{epochs} | train loss {tot / len(Xi):.3f} | val acc {va:.4f} | {time.time() - t0:.0f}s")
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    t1 = time.time(); probs = predict_hf(model, Ti, Tm); infer_time = time.time() - t1
    return finalize(name, "pretrained", model, hist, best_epoch, best_val, probs, train_time, infer_time, count_params(model, True))


# Experiment C: DistilBERT, frozen (256)
results["C: DistilBERT, frozen (256)"] = train_hf("C: DistilBERT, frozen (256)", "distilbert-base-uncased", 256, 3, True)


# Experiment D: DistilBERT fine-tuned (128)
results["D: DistilBERT fine-tuned (128)"] = train_hf("D: DistilBERT fine-tuned (128)", "distilbert-base-uncased", 128, 3, False)


# Experiment E: DistilBERT fine-tuned (256)
results["E: DistilBERT fine-tuned (256)"] = train_hf("E: DistilBERT fine-tuned (256)", "distilbert-base-uncased", 256, 3, False)


# Experiment F: BERT-base fine-tuned (256)
if RUN_BERT_BASE:
    results["F: BERT-base fine-tuned (256)"] = train_hf("F: BERT-base fine-tuned (256)", "bert-base-uncased", 256, 2, False)
else:
    print("Skipped (RUN_BERT_BASE = False).")


# ======================================================================
# 6. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Params": f"{r['params']:,}",
        "Trainable": f"{r['trainable']:,}",
        "Best epoch": r["best_epoch"],
        "Val acc (%)": round(r["best_val_acc"] * 100, 2),
        "Test acc (%)": round(r["test_acc"] * 100, 2),
        "Test F1 (%)": round(r["test_f1"] * 100, 2),
        "Train time (s)": round(r["time"]),
        "Inference, 25k reviews (s)": round(r["infer_time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(f"Reference: Day 6 GRU test accuracy = {DAY6_GRU_TEST_ACC}%")
print(summary.to_string(index=False))

# Learning curves (validation accuracy per epoch) and training loss
names = list(results)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
for n in names:
    h = results[n]["hist"]; ep = np.arange(1, len(h["val_acc"]) + 1)
    ax[0].plot(ep, np.array(h["val_acc"]) * 100, marker="o", ms=4, label=n)
    ax[1].plot(ep, h["train_loss"], marker="o", ms=4, label=n)
ax[0].set_title("Validation accuracy (%)"); ax[1].set_title("Training loss")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=7)
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# Test accuracy with bootstrap 95% CI, and accuracy by review length
correct = {n: ((results[n]["test_probs"] > 0.5).astype(int) == yte_np).astype(float) for n in names}
rng = np.random.RandomState(SEED)
boot = [rng.randint(0, len(yte_np), len(yte_np)) for _ in range(1000)]
ci = {n: np.percentile([correct[n][b].mean() * 100 for b in boot], [2.5, 97.5]) for n in names}

fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
accs = [results[n]["test_acc"] * 100 for n in names]
y = np.arange(len(names))[::-1]
ax[0].barh(y, accs, xerr=[[a - ci[n][0] for a, n in zip(accs, names)], [ci[n][1] - a for a, n in zip(accs, names)]],
           capsize=3, color=["tab:gray" if results[n]["family"] == "scratch" else "tab:blue" for n in names])
ax[0].axvline(DAY6_GRU_TEST_ACC, color="red", ls="--", label=f"Day 6 GRU ({DAY6_GRU_TEST_ACC}%)")
for yy, a in zip(y, accs):
    ax[0].text(a + 0.15, yy, f"{a:.2f}", va="center", fontsize=9)
ax[0].set_yticks(y); ax[0].set_yticklabels([n.split(":")[0] for n in names])
ax[0].set_xlim(min(accs) - 4, max(accs) + 2.5); ax[0].set_xlabel("test accuracy (%) with bootstrap 95% CI")
ax[0].legend(loc="lower right", fontsize=8); ax[0].set_title("Test accuracy (gray = random init, blue = pretrained)")

bins = [(0, 100), (100, 200), (200, 400), (400, 10**9)]
labels_b = ["< 100", "100-199", "200-399", ">= 400"]
w = 0.8 / len(names)
for k, n in enumerate(names):
    vals = [correct[n][(Ote_np >= lo) & (Ote_np < hi)].mean() * 100 for lo, hi in bins]
    ax[1].bar(np.arange(len(bins)) + (k - (len(names) - 1) / 2) * w, vals, w, label=n.split(":")[0] + ": " + n.split(":")[1].strip())
ax[1].set_xticks(np.arange(len(bins))); ax[1].set_xticklabels(labels_b); ax[1].set_xlabel("review length (tokens)")
ax[1].set_ylim(max(40, min(accs) - 15), 100); ax[1].set_ylabel("test accuracy (%)")
ax[1].set_title("Test accuracy by review length"); ax[1].legend(fontsize=6.5, ncol=2)
plt.tight_layout()
save_fig("accuracy_comparison.png")
plt.show()

for n in names:
    print(f"{n:<36} test acc {results[n]['test_acc'] * 100:.2f}  95% CI [{ci[n][0]:.2f}, {ci[n][1]:.2f}]")
print("reviews per length bucket:", {l: int(((Ote_np >= lo) & (Ote_np < hi)).sum()) for l, (lo, hi) in zip(labels_b, bins)})

# Paired bootstrap of accuracy differences (same resampled test reviews for both models)
def paired(a, b):
    d = np.array([(correct[a][i].mean() - correct[b][i].mean()) * 100 for i in boot])
    print(f"{a.split(':')[0]} - {b.split(':')[0]}: {d.mean():+.2f} pp  95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]"
          f"  | {a.split(':')[0]} better in {np.mean(d > 0):.0%} of resamples")

keys = {k: [n for n in names if n.startswith(k + ":")] for k in "ABCDEF"}
get = lambda k: keys[k][0] if keys[k] else None
pairs = [("A", "B"), ("E", "A"), ("E", "C"), ("E", "D"), ("F", "E")]
for a, b in pairs:
    if get(a) and get(b):
        paired(get(a), get(b))

# Calibration: how trustworthy are the predicted probabilities? (Day 6: the GRU had mean confidence 0.872 on its mistakes)
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfectly calibrated")
for n in names:
    p = results[n]["test_probs"]; pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p); ok = (pred == yte_np).astype(float)
    edges = np.linspace(0.5, 1.0, 11); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() >= 30:
            xs.append(conf[m].mean()); ys.append(ok[m].mean())
    ax.plot(xs, ys, marker="o", ms=3, label=n.split(":")[0])
    print(f"{n:<36} ECE {ece_score(conf, ok):.3f} | mean confidence: correct {conf[ok == 1].mean():.3f}, wrong {conf[ok == 0].mean():.3f}")
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram (test set)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
save_fig("calibration.png")
plt.show()


# ======================================================================
# Error Analysis (best model)
# ======================================================================
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best = results[best_name]
probs = best["test_probs"]; pred = (probs > 0.5).astype(int)
print("Best config by validation accuracy:", best_name, f"| test acc {best['test_acc'] * 100:.2f}%")
cm = confusion_matrix(yte_np, pred); tn, fp, fn, tp = cm.ravel()
print(f"confusion matrix [[TN FP] [FN TP]]: {cm.tolist()} | precision {tp / (tp + fp):.3f} | recall {tp / (tp + fn):.3f}")

conf = np.where(pred == 1, probs, 1 - probs)
wrong = np.where(pred != yte_np)[0]
worst = wrong[np.argsort(-conf[wrong])][:5]
for i in worst:
    t = te_texts[i].replace("<br />", " ")
    print(f"\n[true: {'positive' if yte_np[i] else 'negative'} | predicted: {'positive' if pred[i] else 'negative'} "
          f"| confidence {conf[i]:.3f} | {Ote_np[i]} tokens]\n  {t[:300]}...")


# ======================================================================
# What Do the Attention Heads Look At?
# ======================================================================
a_name, b_name = get("A"), get("B")
if a_name and b_name:
    cand = np.where((Ote_np >= 14) & (Ote_np <= 45))[0]
    if len(cand) == 0:
        cand = np.argsort(Ote_np)[:20]
    i = int(np.random.RandomState(SEED).choice(cand))
    n_tok = int(Lte_np[i]); toks = [itos[t] for t in Xte_np[i, :n_tok]]
    fig, axes = plt.subplots(2, N_HEADS, figsize=(4.4 * N_HEADS, 9.2))
    for row, nm in enumerate([a_name, b_name]):
        mdl = results[nm]["model"].eval()
        with torch.no_grad():
            _, attns = mdl(Xte[i:i + 1, :n_tok], return_attn=True)
        A1 = attns[0][0].cpu().numpy()                              # layer 1: (heads, L, L)
        for h in range(N_HEADS):
            ax = axes[row, h]
            ax.imshow(A1[h, :n_tok, :n_tok], cmap="viridis")
            ax.set_xticks(range(n_tok)); ax.set_xticklabels(toks, rotation=90, fontsize=6)
            ax.set_yticks(range(n_tok)); ax.set_yticklabels(toks, fontsize=6)
            ax.set_title(f"{'with' if row == 0 else 'without'} positions | layer 1, head {h + 1}", fontsize=9)
    plt.suptitle(f"First-layer attention on one review (row = word looking, column = word looked at)")
    plt.tight_layout()
    save_fig("attention_own_transformer.png")
    plt.show()

try:
    e_name = get("E")
    tk = AutoTokenizer.from_pretrained("bert-base-uncased")
    try:
        viz = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, attn_implementation="eager")
    except TypeError:
        viz = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    viz.load_state_dict({k: v.cpu() for k, v in results[e_name]["model"].state_dict().items()})
    viz = viz.to(DEVICE).eval()

    probs_e = results[e_name]["test_probs"]; pred_e = (probs_e > 0.5).astype(int)
    rs = np.random.RandomState(SEED)
    short = np.where(Ote_np < 150)[0]
    picks = list(rs.choice(short[(pred_e[short] == yte_np[short]) & (yte_np[short] == 1)], 2, replace=False)) + \
            list(rs.choice(short[(pred_e[short] == yte_np[short]) & (yte_np[short] == 0)], 2, replace=False))
    for i in picks:
        enc = tk(te_texts[i].replace("<br />", " "), truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = viz(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_attentions=True)
        att = out.attentions[-1][0].mean(0)[0].cpu().numpy()          # last layer, mean over heads, from [CLS] to every token
        toks = tk.convert_ids_to_tokens(enc["input_ids"][0].cpu())
        order = np.argsort(-att)[:8]
        print(f"[true {'pos' if yte_np[i] else 'neg'} | P(pos) = {probs_e[i]:.2f}]  [CLS] attends most to: " +
              ", ".join(f"{toks[k]}({att[k]:.2f})" for k in order))
except Exception as ex:
    print("Attention visualization skipped:", repr(ex)[:200])

# Save the best model (state dict) and the vocabulary used by the from-scratch models
torch.save({k: v.cpu() for k, v in best["model"].state_dict().items()}, OUT / "best_model.pth")
(OUT / "vocab.json").write_text(json.dumps(itos))
print("Saved", best_name, "to", OUT)


# ======================================================================
# 7. Analysis
# ======================================================================