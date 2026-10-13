"""
Day 6: RNN, LSTM and GRU for IMDB sentiment analysis (PyTorch).

Script version of lstm_gru_imdb.ipynb. Includes hand-written LSTM/GRU cells verified against PyTorch,
five experiments (RNN, LSTM, GRU, BiLSTM, BiGRU), a gradient-through-time analysis and error analysis.
Downloads IMDB through Hugging Face `datasets` (internet needed) or falls back to a Kaggle CSV.
A GPU is recommended (about 15-30 minutes on a Kaggle T4).
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
import os, re, copy, time, random
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
VOCAB_SIZE = 20000     # most frequent training words (plus <pad> and <unk>)
MAX_LEN = 400          # reviews are truncated to their first 400 tokens
EMB_DIM = 128
HIDDEN = 128
DROPOUT = 0.4
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
CLIP = 1.0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Fallback: Kaggle CSV with columns 'review' and 'sentiment'
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
            print(f"WARNING: using {p} with a RANDOM split; reviews of the same movie can appear on both sides, "
                  f"so the scores are optimistic.")
            return "csv", [texts[i] for i in tr], y[tr].tolist(), [texts[i] for i in te], y[te].tolist()
    raise RuntimeError("Dataset not found. Turn Internet ON (Settings) so it can be downloaded, or add the Kaggle dataset "
                       "'IMDB Dataset of 50K Movie Reviews' with '+ Add Input'.")

source, tr_texts_all, tr_labels_all, te_texts, te_labels = load_imdb()
print("data source:", source, "| train+val reviews:", len(tr_texts_all), "| test reviews:", len(te_texts))

# train / validation split
tr_idx, va_idx = train_test_split(np.arange(len(tr_texts_all)), test_size=0.10, stratify=tr_labels_all, random_state=SEED)
train_texts = [tr_texts_all[i] for i in tr_idx]; train_labels = [tr_labels_all[i] for i in tr_idx]
val_texts   = [tr_texts_all[i] for i in va_idx]; val_labels   = [tr_labels_all[i] for i in va_idx]

# leakage check: exact duplicates of training reviews in validation / test
train_set = set(train_texts)
keep_v = [i for i, t in enumerate(val_texts) if t not in train_set]
keep_t = [i for i, t in enumerate(te_texts) if t not in train_set]
print(f"duplicates of training reviews removed: {len(val_texts) - len(keep_v)} from validation, {len(te_texts) - len(keep_t)} from test")
val_texts = [val_texts[i] for i in keep_v]; val_labels = [val_labels[i] for i in keep_v]
te_texts  = [te_texts[i] for i in keep_t];  te_labels  = [te_labels[i] for i in keep_t]
print(f"final sizes: train {len(train_texts):,} | val {len(val_texts):,} | test {len(te_texts):,} | "
      f"positive share: train {np.mean(train_labels):.1%}, test {np.mean(te_labels):.1%}")

# Tokenization, vocabulary (from the training reviews only), encoding
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")
def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("<br />", " "))

counter = Counter(tok for t in train_texts for tok in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common(VOCAB_SIZE) if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}
print("vocabulary size:", len(itos), "| most frequent:", itos[2:12])

def encode(texts):
    n = len(texts)
    X = np.zeros((n, MAX_LEN), np.int64); L = np.zeros(n, np.int64); O = np.zeros(n, np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tokenize(t)] or [1]
        O[i] = len(ids)
        ids = ids[:MAX_LEN]; L[i] = len(ids); X[i, :L[i]] = ids
    return X, L, O

Xtr_np, Ltr_np, Otr_np = encode(train_texts)
Xva_np, Lva_np, Ova_np = encode(val_texts)
Xte_np, Lte_np, Ote_np = encode(te_texts)
ytr_np, yva_np, yte_np = np.array(train_labels), np.array(val_labels), np.array(te_labels)

unk_rate = (Xva_np == 1).sum() / Lva_np.sum()
print(f"unknown-token rate on validation reviews: {unk_rate:.2%} | reviews longer than {MAX_LEN} tokens (truncated): "
      f"{(Otr_np > MAX_LEN).mean():.1%} of train, {(Ote_np > MAX_LEN).mean():.1%} of test")

to_t = lambda a, dt=None: torch.as_tensor(a, device=DEVICE, dtype=dt)
Xtr, Ltr, ytr = to_t(Xtr_np), to_t(Ltr_np), to_t(ytr_np, torch.float32)
Xva, Lva, yva = to_t(Xva_np), to_t(Lva_np), to_t(yva_np, torch.float32)
Xte, Lte, yte = to_t(Xte_np), to_t(Lte_np), to_t(yte_np, torch.float32)

plt.figure(figsize=(7, 3.5))
plt.hist(np.clip(Otr_np, 0, 1200), bins=60, color="tab:blue")
plt.axvline(MAX_LEN, color="red", ls="--", label=f"max length = {MAX_LEN}")
plt.xlabel("review length (tokens, clipped at 1200)"); plt.ylabel("reviews"); plt.legend()
plt.title("Review length distribution (training set)")
plt.tight_layout()
save_fig("length_distribution.png")
plt.show()


# ======================================================================
# 3. LSTM and GRU Cells, Implemented by Hand
# ======================================================================
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = 1.0 / hidden_size ** 0.5
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))

    def forward(self, x, state):
        h, c = state
        gates = x @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_new = f * c + i * g                  # additive cell update
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = 1.0 / hidden_size ** 0.5
        self.W_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size).uniform_(-k, k))
        self.W_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size).uniform_(-k, k))
        self.b_ih = nn.Parameter(torch.empty(3 * hidden_size).uniform_(-k, k))
        self.b_hh = nn.Parameter(torch.empty(3 * hidden_size).uniform_(-k, k))

    def forward(self, x, h):
        gi = x @ self.W_ih.T + self.b_ih
        gh = h @ self.W_hh.T + self.b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h             # leaky integration of the previous state


# Numerical check against PyTorch's reference cells (same weights, same inputs)
torch.manual_seed(0)
I, H, B = 16, 32, 5
x, h, c = torch.randn(B, I), torch.randn(B, H), torch.randn(B, H)

ref, mine = nn.LSTMCell(I, H), MyLSTMCell(I, H)
with torch.no_grad():
    mine.W_ih.copy_(ref.weight_ih); mine.W_hh.copy_(ref.weight_hh); mine.b_ih.copy_(ref.bias_ih); mine.b_hh.copy_(ref.bias_hh)
(h1, c1), (h2, c2) = ref(x, (h, c)), mine(x, (h, c))
d_lstm = max((h1 - h2).abs().max().item(), (c1 - c2).abs().max().item())

ref, mine = nn.GRUCell(I, H), MyGRUCell(I, H)
with torch.no_grad():
    mine.W_ih.copy_(ref.weight_ih); mine.W_hh.copy_(ref.weight_hh); mine.b_ih.copy_(ref.bias_ih); mine.b_hh.copy_(ref.bias_hh)
d_gru = (ref(x, h) - mine(x, h)).abs().max().item()

print(f"LSTM cell: max abs difference vs nn.LSTMCell = {d_lstm:.2e} ->", "matches" if d_lstm < 1e-5 else "MISMATCH, check the gate order")
print(f"GRU  cell: max abs difference vs nn.GRUCell  = {d_gru:.2e} ->", "matches" if d_gru < 1e-5 else "MISMATCH, check the equations")


# ======================================================================
# 4. Model
# ======================================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, rnn_type="LSTM", emb_dim=EMB_DIM, hidden=HIDDEN, bidirectional=False, dropout=DROPOUT):
        super().__init__()
        self.rnn_type, self.bi = rnn_type, bidirectional
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        extra = {"nonlinearity": "tanh"} if rnn_type == "RNN" else {}
        self.rnn = cls(emb_dim, hidden, batch_first=True, bidirectional=bidirectional, **extra)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x, lengths, return_emb=False):
        emb = self.emb(x)
        if return_emb:
            emb.retain_grad()                  # used for gradient / saliency analysis
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.rnn_type == "LSTM":
            h = h[0]                           # (h_n, c_n) -> h_n
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bi else h[-1]     # final state(s) after the last real token
        logit = self.fc(self.dropout(h)).squeeze(1)
        return (logit, emb) if return_emb else logit

def count_params(m):
    return sum(p.numel() for p in m.parameters())

for name, t, bi in [("RNN", "RNN", False), ("LSTM", "LSTM", False), ("GRU", "GRU", False),
                    ("BiLSTM", "LSTM", True), ("BiGRU", "GRU", True)]:
    print(f"{name:7s} {count_params(SentimentRNN(len(itos), t, bidirectional=bi)):,} parameters")


# ======================================================================
# 5. Training Setup
# ======================================================================
@torch.no_grad()
def predict(model, X, L, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs], L[i:i + bs])).cpu())
    return torch.cat(out).numpy()

def run_epoch(model, X, L, y, optimizer=None):
    training = optimizer is not None
    model.train(training)
    idx = torch.randperm(len(X), device=DEVICE) if training else torch.arange(len(X), device=DEVICE)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(idx), BATCH_SIZE):
            j = idx[i:i + BATCH_SIZE]
            logits = model(X[j], L[j])
            loss = F.binary_cross_entropy_with_logits(logits, y[j])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            total_loss += loss.item() * len(j)
            correct += ((logits > 0).float() == y[j]).sum().item()
    return total_loss / len(X), correct / len(X)

def train_experiment(name, rnn_type, bidirectional):
    print(f"\n=== {name} | {rnn_type} | bidirectional={bidirectional} ===")
    torch.manual_seed(SEED)
    model = SentimentRNN(len(itos), rnn_type, bidirectional=bidirectional).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, Xtr, Ltr, ytr, optimizer)
        vl, va = run_epoch(model, Xva, Lva, yva)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        if va > best_val:
            best_val, best_epoch = va, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {ep}/{EPOCHS} | train loss {tl:.3f} acc {ta:.4f} | val loss {vl:.3f} acc {va:.4f} | {time.time() - t0:.0f}s")
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    probs = predict(model, Xte, Lte)
    pred = (probs > 0.5).astype(int)
    test_acc = float((pred == yte_np).mean()); test_f1 = f1_score(yte_np, pred)
    print(f"-> best epoch {best_epoch} | val acc {best_val:.4f} | test acc {test_acc:.4f} | F1 {test_f1:.4f} | {elapsed:.0f}s")
    return dict(model=model, hist=hist, rnn_type=rnn_type, bi=bidirectional, params=count_params(model),
                best_epoch=best_epoch, best_val_acc=best_val, test_acc=test_acc, test_f1=test_f1,
                test_probs=probs, time=elapsed)

results = {}


# Experiment A: RNN
results["A: RNN"] = train_experiment("A: RNN", "RNN", False)


# Experiment B: LSTM
results["B: LSTM"] = train_experiment("B: LSTM", "LSTM", False)


# Experiment C: GRU
results["C: GRU"] = train_experiment("C: GRU", "GRU", False)


# Experiment D: BiLSTM
results["D: BiLSTM"] = train_experiment("D: BiLSTM", "LSTM", True)


# Experiment E: BiGRU
results["E: BiGRU"] = train_experiment("E: BiGRU", "GRU", True)


# ======================================================================
# 6. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Params": f"{r['params']:,}",
        "Best epoch": r["best_epoch"],
        "Val acc (%)": round(r["best_val_acc"] * 100, 2),
        "Test acc (%)": round(r["test_acc"] * 100, 2),
        "Test F1 (%)": round(r["test_f1"] * 100, 2),
        "Time (s)": round(r["time"]),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "results_summary.csv", index=False)
print(summary.to_string(index=False))

# Learning curves
names = list(results)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for n in names:
    h = results[n]["hist"]; ep = np.arange(1, EPOCHS + 1)
    ax[0].plot(ep, np.array(h["val_acc"]) * 100, marker="o", ms=3, label=n)
    ax[1].plot(ep, h["train_loss"], marker="o", ms=3, label=n)
    ax[2].plot(ep, h["val_loss"], marker="o", ms=3, label=n)
ax[0].set_title("Validation accuracy (%)"); ax[1].set_title("Training loss"); ax[2].set_title("Validation loss")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# Test accuracy overall and by review length (length = original number of tokens, before truncation)
bins = [(0, 100), (100, 200), (200, 400), (400, 10**9)]
labels_b = ["< 100", "100-199", "200-399", ">= 400 (truncated)"]
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})

accs = [results[n]["test_acc"] * 100 for n in names]
bars = ax[0].bar([n.split(":")[0] for n in names], accs, color=["tab:gray", "tab:blue", "tab:green", "tab:purple", "tab:red"])
for b, a_ in zip(bars, accs):
    ax[0].text(b.get_x() + b.get_width() / 2, a_ + 0.3, f"{a_:.2f}", ha="center", fontsize=9)
ax[0].set_ylim(max(40, min(accs) - 5), min(100, max(accs) + 3)); ax[0].set_title("Test accuracy (%)")
ax[0].set_xlabel("A: RNN | B: LSTM | C: GRU | D: BiLSTM | E: BiGRU", fontsize=8)

w = 0.16
for k, n in enumerate(names):
    pred = (results[n]["test_probs"] > 0.5).astype(int)
    vals = [((pred == yte_np)[(Ote_np >= lo) & (Ote_np < hi)]).mean() * 100 for lo, hi in bins]
    ax[1].bar(np.arange(len(bins)) + (k - 2) * w, vals, w, label=n)
ax[1].set_xticks(np.arange(len(bins))); ax[1].set_xticklabels(labels_b)
ax[1].set_ylim(max(40, min(accs) - 15), 100); ax[1].set_ylabel("test accuracy (%)")
ax[1].set_title("Test accuracy by review length (tokens)"); ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout()
save_fig("accuracy_by_length.png")
plt.show()

print("reviews per length bucket:", {l: int(((Ote_np >= lo) & (Ote_np < hi)).sum()) for l, (lo, hi) in zip(labels_b, bins)})


# ======================================================================
# 7. Do Gradients Vanish Through Time?
# ======================================================================
def grad_profile(model, X, L):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.backends.cudnn.flags(enabled=False):
        logit, emb = model(X, L, return_emb=True)
        logit.sum().backward()
    return emb.grad.norm(dim=-1).mean(0).cpu().numpy()          # mean over reviews, per position

long_ids = np.where(Ote_np >= MAX_LEN)[0][:256]
print(f"{len(long_ids)} long test reviews used")
long_t = torch.as_tensor(long_ids, device=DEVICE)
Xl, Ll = Xte[long_t], Lte[long_t]

plt.figure(figsize=(8.5, 4.5))
ratios = {}
for n in ["A: RNN", "B: LSTM", "C: GRU"]:
    g = grad_profile(results[n]["model"], Xl, Ll) + 1e-12
    g = g / g[-10:].mean()
    plt.semilogy(np.arange(MAX_LEN)[::-1], g, label=n)
    ratios[n] = g[:50].mean() / g[-50:].mean()
plt.xlabel("distance from the end of the review (tokens)"); plt.ylabel("gradient norm, relative to the last 10 tokens (log scale)")
plt.title("Influence of each word on the output (trained models)")
plt.legend(); plt.grid(alpha=0.3)
save_fig("gradient_through_time.png")
plt.show()

for n, r_ in ratios.items():
    print(f"{n}: gradient at the first 50 tokens / last 50 tokens = {r_:.4f}")


# ======================================================================
# Error Analysis (best model)
# ======================================================================
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best = results[best_name]
probs = best["test_probs"]; pred = (probs > 0.5).astype(int)
print("Best config by validation accuracy:", best_name, f"| test acc {best['test_acc']*100:.2f}%")

cm = confusion_matrix(yte_np, pred)
tn, fp, fn, tp = cm.ravel()
print(f"confusion matrix [[TN FP] [FN TP]]: {cm.tolist()} | precision {tp / (tp + fp):.3f} | recall {tp / (tp + fn):.3f}")

# Confidence of correct vs wrong predictions
conf = np.where(pred == 1, probs, 1 - probs)
print(f"mean confidence: correct {conf[pred == yte_np].mean():.3f} | wrong {conf[pred != yte_np].mean():.3f}")

# Most confident mistakes
wrong = np.where(pred != yte_np)[0]
worst = wrong[np.argsort(-conf[wrong])][:5]
for i in worst:
    t = te_texts[i].replace("<br />", " ")
    print(f"\n[true: {'positive' if yte_np[i] else 'negative'} | predicted: {'positive' if pred[i] else 'negative'} "
          f"| confidence {conf[i]:.3f} | {Ote_np[i]} tokens]\n  {t[:350]}...")

def saliency(model, i):
    model.eval(); model.zero_grad(set_to_none=True)
    with torch.backends.cudnn.flags(enabled=False):
        logit, emb = model(Xte[i:i + 1], Lte[i:i + 1], return_emb=True)
        logit.sum().backward()
    n = int(Lte_np[i])
    return emb.grad.norm(dim=-1)[0, :n].cpu().numpy(), torch.sigmoid(logit).item()

rs = np.random.RandomState(SEED)
correct_pos = rs.choice(np.where((pred == yte_np) & (yte_np == 1))[0], 2, replace=False)
correct_neg = rs.choice(np.where((pred == yte_np) & (yte_np == 0))[0], 2, replace=False)
examples = list(correct_pos) + list(correct_neg) + list(worst[:2])

for i in examples:
    s, p_pos = saliency(best["model"], i)
    ids = Xte_np[i, :len(s)]
    order = np.argsort(-s)[:10]
    top = ", ".join(f"{itos[ids[k]]}({s[k] / s.max():.2f})" for k in order)
    status = "correct" if pred[i] == yte_np[i] else "WRONG"
    print(f"[true {'pos' if yte_np[i] else 'neg'} | P(pos) = {p_pos:.2f} | {status}]  top words: {top}")

# Save the best model and the vocabulary
import json
torch.save(best["model"].state_dict(), OUT / "sentiment_rnn_best.pth")
(OUT / "vocab.json").write_text(json.dumps(itos))
print("Saved to", OUT)


# ======================================================================
# 8. Analysis
# ======================================================================