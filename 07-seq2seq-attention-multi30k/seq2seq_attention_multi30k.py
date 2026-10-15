"""
Day 7: Seq2Seq with attention for English-German translation on Multi30k (PyTorch).

Script version of seq2seq_attention_multi30k.ipynb. Trains an encoder-decoder without attention and with
three attention score functions (additive, general, scaled dot-product), compares teacher-forcing ratios,
and reports BLEU (own implementation, checked against SacreBLEU), BLEU by length with bootstrap confidence
intervals, example translations and attention heatmaps.
Downloads Multi30k through Hugging Face `datasets` (internet needed) or reads local Multi30k text files.
A GPU is recommended (about 35-50 minutes on a Kaggle T4).
"""
import matplotlib
matplotlib.use("Agg")   # headless: figures are saved to disk, not shown


# ======================================================================
# 1. Background
# ======================================================================

# Experiment design
import os, re, math, copy, time, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

SEED = 42
MAX_LEN = 50           # maximum number of tokens per sentence (without <sos>/<eos>)
MIN_FREQ = 2           # words seen fewer times in the training set become <unk>
EMB_DIM, ENC_HID, DEC_HID = 256, 256, 512
DROPOUT = 0.4
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 12
CLIP = 1.0
PAD, UNK, SOS, EOS = 0, 1, 2, 3

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
def load_multi30k():
    try:
        from datasets import load_dataset
        ds = load_dataset("bentrevett/multi30k")
        get = lambda split: (list(ds[split]["en"]), list(ds[split]["de"]))
        return get("train"), get("validation"), get("test")
    except Exception as e:
        print("Could not load Multi30k from Hugging Face:", repr(e)[:150])
    # Fallback: plain-text files found under /kaggle/input
    def find(*names):
        for n in names:
            hits = sorted(Path("/kaggle/input").rglob(n))
            if hits:
                return hits[0]
        return None
    read = lambda p: [l.rstrip("\n") for l in open(p, encoding="utf-8")]
    files = {
        "train": (find("train.en"), find("train.de")),
        "val":   (find("val.en", "valid.en", "validation.en"), find("val.de", "valid.de", "validation.de")),
        "test":  (find("test_2016_flickr.en", "test.en"), find("test_2016_flickr.de", "test.de")),
    }
    if all(a is not None and b is not None for a, b in files.values()):
        print("Using local Multi30k files:", {k: (str(a.name), str(b.name)) for k, (a, b) in files.items()})
        return tuple((read(a), read(b)) for a, b in files.values())
    raise RuntimeError("Multi30k not found. Turn Internet ON (Settings) so it can be downloaded, or add a Kaggle dataset "
                       "with train.en/train.de, val.en/val.de and test_2016_flickr.en/test_2016_flickr.de via '+ Add Input'.")

(tr_src, tr_tgt), (va_src, va_tgt), (te_src, te_tgt) = load_multi30k()
print(f"sentence pairs: train {len(tr_src):,} | val {len(va_src):,} | test {len(te_src):,}")
print("example:", tr_src[0], "->", tr_tgt[0])

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
tok = lambda s: TOKEN_RE.findall(s.lower())

tr_src_t, tr_tgt_t = [tok(s) for s in tr_src], [tok(s) for s in tr_tgt]
va_src_t, va_tgt_t = [tok(s) for s in va_src], [tok(s) for s in va_tgt]
te_src_t, te_tgt_t = [tok(s) for s in te_src], [tok(s) for s in te_tgt]

# drop empty or overly long training pairs
keep = [i for i in range(len(tr_src_t)) if 0 < len(tr_src_t[i]) <= MAX_LEN and 0 < len(tr_tgt_t[i]) <= MAX_LEN]
print(f"training pairs kept: {len(keep):,} of {len(tr_src_t):,}")
tr_src_t, tr_tgt_t = [tr_src_t[i] for i in keep], [tr_tgt_t[i] for i in keep]

def build_vocab(token_lists):
    c = Counter(w for t in token_lists for w in t)
    return ["<pad>", "<unk>", "<sos>", "<eos>"] + [w for w, n in c.most_common() if n >= MIN_FREQ]

itos_src, itos_tgt = build_vocab(tr_src_t), build_vocab(tr_tgt_t)
stoi_src, stoi_tgt = {w: i for i, w in enumerate(itos_src)}, {w: i for i, w in enumerate(itos_tgt)}
print(f"vocabulary: English {len(itos_src):,} | German {len(itos_tgt):,}")
print(f"average length (tokens): English {np.mean([len(t) for t in tr_src_t]):.1f} | German {np.mean([len(t) for t in tr_tgt_t]):.1f}")

def encode(token_lists, stoi, truncate=MAX_LEN):
    W = MAX_LEN + 2
    X = np.zeros((len(token_lists), W), np.int64); L = np.zeros(len(token_lists), np.int64)
    for i, t in enumerate(token_lists):
        ids = [SOS] + [stoi.get(w, UNK) for w in t[:truncate]] + [EOS]
        X[i, :len(ids)] = ids; L[i] = len(ids)
    return X, L

to_t = lambda a: torch.as_tensor(a, device=DEVICE)
Xs_tr, Ls_tr = map(to_t, encode(tr_src_t, stoi_src)); Xt_tr, Lt_tr = map(to_t, encode(tr_tgt_t, stoi_tgt))
Xs_va, Ls_va = map(to_t, encode(va_src_t, stoi_src)); Xt_va, Lt_va = map(to_t, encode(va_tgt_t, stoi_tgt))
Xs_te, Ls_te = map(to_t, encode(te_src_t, stoi_src)); Xt_te, Lt_te = map(to_t, encode(te_tgt_t, stoi_tgt))

unk_rate = lambda X: (X == UNK).sum().item() / ((X == UNK) | (X > EOS)).sum().item()
print(f"unknown-word rate: train source {unk_rate(Xs_tr):.2%} | val source {unk_rate(Xs_va):.2%} | val target {unk_rate(Xt_va):.2%}")


# ======================================================================
# 3. BLEU (my own implementation)
# ======================================================================
def bleu_stats(ref, hyp, max_n=4):
    """[hyp_len, ref_len, clipped_1..4, total_1..4] for one sentence pair."""
    s = np.zeros(2 + 2 * max_n)
    s[0], s[1] = len(hyp), len(ref)
    for n in range(1, max_n + 1):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        s[1 + n] = sum(min(c, r[g]) for g, c in h.items())
        s[1 + max_n + n] = sum(h.values())
    return s

def bleu_from_stats(s, max_n=4):
    hyp_len, ref_len = s[0], s[1]
    clipped, total = s[2:2 + max_n], s[2 + max_n:]
    if (clipped == 0).any() or (total == 0).any():
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(np.mean(np.log(clipped / total)))

def corpus_bleu(refs, hyps):
    return bleu_from_stats(sum(bleu_stats(r, h) for r, h in zip(refs, hyps)))

# sanity checks
ref = "the cat sat on the mat".split()
print("identical sentence ->", round(corpus_bleu([ref], [ref]), 2), "(expected 100.0)")
print("hypothesis one word short ->", round(corpus_bleu([ref], [ref[:5]]), 2), "(expected 81.87 = 100 * exp(1 - 6/5))")


# ======================================================================
# 4. Model
# ======================================================================
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, attn_type="additive"):
        super().__init__()
        self.attn_type = attn_type
        enc_dim = 2 * ENC_HID
        self.src_emb = nn.Embedding(src_vocab, EMB_DIM, padding_idx=PAD)
        self.encoder = nn.GRU(EMB_DIM, ENC_HID, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(enc_dim, DEC_HID)
        self.tgt_emb = nn.Embedding(tgt_vocab, EMB_DIM, padding_idx=PAD)
        self.use_attn = attn_type != "none"
        if attn_type == "additive":
            self.W_s = nn.Linear(DEC_HID, DEC_HID, bias=False)
            self.W_h = nn.Linear(enc_dim, DEC_HID, bias=False)
            self.v = nn.Linear(DEC_HID, 1, bias=False)
        elif attn_type == "general":
            self.W_a = nn.Linear(enc_dim, DEC_HID, bias=False)
        elif attn_type == "dot":
            assert enc_dim == DEC_HID, "scaled dot-product attention needs equal dimensions"
        ctx = enc_dim if self.use_attn else 0
        self.cell = nn.GRUCell(EMB_DIM + ctx, DEC_HID)
        self.out = nn.Linear(DEC_HID + ctx + EMB_DIM, tgt_vocab)
        self.drop = nn.Dropout(DROPOUT)

    def encode(self, src, src_len):
        emb = self.drop(self.src_emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.encoder(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        s0 = torch.tanh(self.bridge(torch.cat([h[-2], h[-1]], dim=1)))
        return H, s0

    def attend(self, s, H, mask):
        if self.attn_type == "additive":
            e = self.v(torch.tanh(self.W_s(s).unsqueeze(1) + self.W_h(H))).squeeze(2)
        elif self.attn_type == "general":
            e = torch.bmm(self.W_a(H), s.unsqueeze(2)).squeeze(2)
        else:  # scaled dot-product
            e = torch.bmm(H, s.unsqueeze(2)).squeeze(2) / math.sqrt(H.size(2))
        e = e.masked_fill(~mask, -1e9)                      # never attend to padding
        a = torch.softmax(e, dim=1)
        return torch.bmm(a.unsqueeze(1), H).squeeze(1), a   # context vector, attention weights

    def step(self, y_prev, s, H, mask):
        emb = self.drop(self.tgt_emb(y_prev))
        if self.use_attn:
            ctx, a = self.attend(s, H, mask)
            s = self.cell(torch.cat([emb, ctx], dim=1), s)
            logits = self.out(self.drop(torch.cat([s, ctx, emb], dim=1)))
        else:
            a = None
            s = self.cell(emb, s)
            logits = self.out(self.drop(torch.cat([s, emb], dim=1)))
        return logits, s, a

    def forward(self, src, src_len, tgt, tf_ratio=1.0):
        H, s = self.encode(src, src_len)
        mask = src != PAD
        y, outputs = tgt[:, 0], []
        for t in range(1, tgt.size(1)):
            logits, s, _ = self.step(y, s, H, mask)
            outputs.append(logits)
            y = tgt[:, t] if (tf_ratio >= 1.0 or random.random() < tf_ratio) else logits.argmax(1)
        return torch.stack(outputs, dim=1)                  # (B, T-1, vocab)

    @torch.no_grad()
    def greedy(self, src, src_len, max_len=60):
        self.eval()
        H, s = self.encode(src, src_len)
        mask = src != PAD
        y = torch.full((src.size(0),), SOS, dtype=torch.long, device=src.device)
        ys, attn = [], []
        for _ in range(max_len):
            logits, s, a = self.step(y, s, H, mask)
            y = logits.argmax(1)
            ys.append(y)
            if a is not None:
                attn.append(a)
        return torch.stack(ys, dim=1), (torch.stack(attn, dim=1) if attn else None)


def count_params(m):
    return sum(p.numel() for p in m.parameters())

for t in ["none", "additive", "general", "dot"]:
    print(f"{t:9s} {count_params(Seq2Seq(len(itos_src), len(itos_tgt), t)):,} parameters")

# quick shape check on one small batch
m = Seq2Seq(len(itos_src), len(itos_tgt), "additive").to(DEVICE)
j = torch.arange(4, device=DEVICE)
print("forward output:", tuple(m(Xs_tr[j, :int(Ls_tr[j].max())], Ls_tr[j], Xt_tr[j, :int(Lt_tr[j].max())]).shape),
      "(batch, target length - 1, target vocabulary)")


# ======================================================================
# 5. Training Setup
# ======================================================================
def decode_ids(ids, itos):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i not in (PAD, SOS):
            out.append(itos[i])
    return out

@torch.no_grad()
def translate(model, Xs, Ls, bs=256):
    """Greedy translations (list of token lists) and the attention weights per batch."""
    model.eval()
    hyps, attns = [], []
    for i in range(0, len(Xs), bs):
        src, sl = Xs[i:i + bs], Ls[i:i + bs]
        src = src[:, :int(sl.max())]
        ys, a = model.greedy(src, sl)
        hyps += [decode_ids(row, itos_tgt) for row in ys.cpu().numpy()]
        attns.append(a)
    return hyps, attns

@torch.no_grad()
def eval_loss(model, Xs, Ls, Xt, Lt, bs=256):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(Xs), bs):
        src, sl, tgt, tl = Xs[i:i + bs], Ls[i:i + bs], Xt[i:i + bs], Lt[i:i + bs]
        src, tgt = src[:, :int(sl.max())], tgt[:, :int(tl.max())]
        out = model(src, sl, tgt, tf_ratio=1.0)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum")
        tot += loss.item(); n += (tgt[:, 1:] != PAD).sum().item()
    return tot / n

def train_experiment(name, attn_type, tf_ratio):
    print(f"\n=== {name} | attention={attn_type} | teacher forcing={tf_ratio} ===")
    torch.manual_seed(SEED); random.seed(SEED)
    model = Seq2Seq(len(itos_src), len(itos_tgt), attn_type).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    hist = {"train_loss": [], "val_loss": [], "val_bleu": []}
    best_bleu, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xs_tr), device=DEVICE)
        tot, ntok = 0.0, 0
        for i in range(0, len(order), BATCH_SIZE):
            j = order[i:i + BATCH_SIZE]
            src, sl, tgt, tl = Xs_tr[j], Ls_tr[j], Xt_tr[j], Lt_tr[j]
            src, tgt = src[:, :int(sl.max())], tgt[:, :int(tl.max())]
            out = model(src, sl, tgt, tf_ratio)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            n = (tgt[:, 1:] != PAD).sum().item(); tot += loss.item() * n; ntok += n
        vl = eval_loss(model, Xs_va, Ls_va, Xt_va, Lt_va)
        hyps, _ = translate(model, Xs_va, Ls_va)
        vb = corpus_bleu(va_tgt_t, hyps)
        hist["train_loss"].append(tot / ntok); hist["val_loss"].append(vl); hist["val_bleu"].append(vb)
        if vb > best_bleu:
            best_bleu, best_epoch = vb, ep
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss {tot / ntok:.3f} | val loss {vl:.3f} (ppl {math.exp(vl):.1f}) | val BLEU {vb:.2f} | {time.time() - t0:.0f}s")
    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    hyps, _ = translate(model, Xs_te, Ls_te)
    stats = np.stack([bleu_stats(r, h) for r, h in zip(te_tgt_t, hyps)])
    test_bleu = bleu_from_stats(stats.sum(0))
    print(f"-> best epoch {best_epoch} | val BLEU {best_bleu:.2f} | test BLEU {test_bleu:.2f} | {elapsed:.0f}s")
    return dict(model=model, hist=hist, attn=attn_type, tf=tf_ratio, params=count_params(model), best_epoch=best_epoch,
                best_val_bleu=best_bleu, test_bleu=test_bleu, test_hyps=hyps, test_stats=stats, time=elapsed)

results = {}


# Experiment A: no attention
results["A: no attention"] = train_experiment("A: no attention", "none", 1.0)


# Experiment B: additive (Bahdanau)
results["B: additive (Bahdanau)"] = train_experiment("B: additive (Bahdanau)", "additive", 1.0)


# Experiment C: general (Luong)
results["C: general (Luong)"] = train_experiment("C: general (Luong)", "general", 1.0)


# Experiment D: scaled dot-product
results["D: scaled dot-product"] = train_experiment("D: scaled dot-product", "dot", 1.0)


# Experiment E: additive, TF 0.5
results["E: additive, TF 0.5"] = train_experiment("E: additive, TF 0.5", "additive", 0.5)


# ======================================================================
# 6. Results
# ======================================================================
rows = []
for name, r in results.items():
    rows.append({
        "Config": name,
        "Attention": r["attn"],
        "Teacher forcing": r["tf"],
        "Params": f"{r['params']:,}",
        "Best epoch": r["best_epoch"],
        "Val BLEU": round(r["best_val_bleu"], 2),
        "Test BLEU": round(r["test_bleu"], 2),
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
    ax[0].plot(ep, h["train_loss"], marker="o", ms=3, label=n)
    ax[1].plot(ep, h["val_loss"], marker="o", ms=3, label=n)
    ax[2].plot(ep, h["val_bleu"], marker="o", ms=3, label=n)
ax[0].set_title("Training loss (per token)"); ax[1].set_title("Validation loss (per token)"); ax[2].set_title("Validation BLEU (greedy)")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
ax[2].legend(fontsize=8)
plt.tight_layout()
save_fig("training_curves_all.png")
plt.show()

# BLEU by source sentence length, with bootstrap 95% confidence intervals (resampling test sentences)
src_len = np.array([len(t) for t in te_src_t])
bins = [(0, 9), (9, 13), (13, 17), (17, 10**9)]
labels_b = ["1-8", "9-12", "13-16", "17+"]
print("test sentences per length bucket:", {l: int(((src_len >= lo) & (src_len < hi)).sum()) for l, (lo, hi) in zip(labels_b, bins)})

rng = np.random.RandomState(SEED)
N_BOOT = 300
boot_idx = [rng.randint(0, len(te_src_t), len(te_src_t)) for _ in range(N_BOOT)]

fig, ax = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
cis = {}
for k, n in enumerate(names):
    S = results[n]["test_stats"]
    bs_ = np.array([bleu_from_stats(S[idx].sum(0)) for idx in boot_idx])
    cis[n] = (np.percentile(bs_, 2.5), np.percentile(bs_, 97.5))
point = [results[n]["test_bleu"] for n in names]
err = [[p - cis[n][0] for p, n in zip(point, names)], [cis[n][1] - p for p, n in zip(point, names)]]
bars = ax[0].bar([n.split(":")[0] for n in names], point, yerr=err, capsize=4, color=["tab:gray", "tab:blue", "tab:green", "tab:purple", "tab:orange"])
for b, p in zip(bars, point):
    ax[0].text(b.get_x() + b.get_width() / 2, p + 0.2, f"{p:.2f}", ha="center", fontsize=9)
ax[0].set_ylim(max(0, min(point) - 6), max(point) + 3); ax[0].set_title("Test BLEU (error bars: bootstrap 95% CI)")
ax[0].set_xlabel("A: none | B: additive | C: general | D: scaled dot | E: additive, TF 0.5", fontsize=8)

w = 0.16
for k, n in enumerate(names):
    S = results[n]["test_stats"]
    vals = [bleu_from_stats(S[(src_len >= lo) & (src_len < hi)].sum(0)) for lo, hi in bins]
    ax[1].bar(np.arange(len(bins)) + (k - 2) * w, vals, w, label=n)
ax[1].set_xticks(np.arange(len(bins))); ax[1].set_xticklabels(labels_b)
ax[1].set_xlabel("source length (tokens)"); ax[1].set_ylabel("test BLEU"); ax[1].set_title("Test BLEU by source sentence length")
ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout()
save_fig("bleu_by_length.png")
plt.show()

for n in names:
    print(f"{n:<28} test BLEU {results[n]['test_bleu']:.2f}  95% CI [{cis[n][0]:.2f}, {cis[n][1]:.2f}]")

# Paired bootstrap: is attention better than no attention? (same resampled test sentences for both models)
best_attn = max([n for n in names if results[n]["attn"] != "none" and results[n]["tf"] == 1.0], key=lambda k: results[k]["best_val_bleu"])
print("best attention model by validation BLEU:", best_attn)
base = results["A: no attention"]["test_stats"]
for n in names[1:]:
    S = results[n]["test_stats"]
    d = np.array([bleu_from_stats(S[idx].sum(0)) - bleu_from_stats(base[idx].sum(0)) for idx in boot_idx])
    print(f"{n:<28} BLEU - BLEU(A) = {d.mean():+.2f}  95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]  "
          f"| better than A in {np.mean(d > 0):.0%} of resamples")

# Example translations (random test sentences)
rs = np.random.RandomState(SEED)
for i in rs.choice(len(te_src_t), 6, replace=False):
    print("EN :", " ".join(te_src_t[i]))
    print("REF:", " ".join(te_tgt_t[i]))
    for n in ["A: no attention", best_attn]:
        print(f"{n.split(':')[0]}  :", " ".join(results[n]["test_hyps"][i]))
    print()

best_model = results[best_attn]["model"]; best_model.eval()
cand = [i for i in range(len(te_src_t)) if 9 <= len(te_src_t[i]) <= 14]
pick = np.random.RandomState(SEED + 3).choice(cand, 3, replace=False)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
for ax_, i in zip(axes, pick):
    src = Xs_te[i:i + 1, :int(Ls_te[i])]
    ys, a = best_model.greedy(src, Ls_te[i:i + 1])
    hyp_ids = ys[0].cpu().numpy()
    n_out = int(np.where(hyp_ids == EOS)[0][0]) + 1 if (hyp_ids == EOS).any() else len(hyp_ids)
    W = a[0, :n_out, :src.size(1)].cpu().numpy()
    ax_.imshow(W, cmap="viridis", aspect="auto")
    ax_.set_xticks(range(src.size(1))); ax_.set_xticklabels([itos_src[t] for t in src[0].cpu().numpy()], rotation=90, fontsize=8)
    ax_.set_yticks(range(n_out)); ax_.set_yticklabels([itos_tgt[t] for t in hyp_ids[:n_out]], fontsize=8)
    ax_.set_xlabel("source (English)"); ax_.set_ylabel("generated (German)")
plt.suptitle(f"Attention weights of {best_attn}")
plt.tight_layout()
save_fig("attention_heatmaps.png")
plt.show()

# How sharp is the attention? (mean of the largest weight per generated word, and entropy relative to uniform)
def attention_sharpness(model):
    model.eval()
    hyps, attns = [], []
    for i in range(0, len(Xs_te), 256):
        src, sl = Xs_te[i:i + 256], Ls_te[i:i + 256]
        src = src[:, :int(sl.max())]
        ys, a = model.greedy(src, sl)
        attns.append((ys.cpu().numpy(), a.cpu().numpy(), sl.cpu().numpy()))
    mx, ent = [], []
    for ys, a, sl in attns:
        for b in range(len(sl)):
            ids = ys[b]; n_out = int(np.where(ids == EOS)[0][0]) + 1 if (ids == EOS).any() else len(ids)
            w = a[b, :n_out, :sl[b]]
            mx.append(w.max(1).mean())
            ent.append((-(w * np.log(w + 1e-12)).sum(1) / np.log(max(sl[b], 2))).mean())
    return np.mean(mx), np.mean(ent)

for n in names:
    if results[n]["attn"] != "none":
        mx, en = attention_sharpness(results[n]["model"])
        print(f"{n:<28} mean max weight {mx:.3f} | normalized entropy {en:.3f}  (1.0 = uniform attention)")

# Output statistics: length ratio, unknown words, repeated words
for n in names:
    hyps = results[n]["test_hyps"]
    hl, rl = sum(len(h) for h in hyps), sum(len(r) for r in te_tgt_t)
    unk = sum(h.count("<unk>") for h in hyps) / max(hl, 1)
    rep = np.mean([any(h[k] == h[k + 1] for k in range(len(h) - 1)) for h in hyps])
    print(f"{n:<28} length ratio (hyp/ref) {hl / rl:.3f} | <unk> in output {unk:.2%} | sentences with an immediately repeated word {rep:.1%}")

# Save the best model and vocabularies
import json
torch.save(results[best_attn]["model"].state_dict(), OUT / "seq2seq_best.pth")
(OUT / "vocab.json").write_text(json.dumps({"src": itos_src, "tgt": itos_tgt}))
print("Saved to", OUT)


# ======================================================================
# 7. Analysis
# ======================================================================