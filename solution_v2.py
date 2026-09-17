# made by - Karthik
import sys
import re
import math
import random
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
SEED = 1234
MAX_LINES = 128
MAX_TOK = 36
MAX_STMT = 160
EMB = 96
SLOT_EMB = 96
NSLOT = 40
CONV1 = 160
CONV2 = 192
DMODEL = 192
NHEAD = 4
NLAYER = 4
FFDIM = 384
DROP = 0.30
FOLDS = 5
EPOCHS = 7
BATCH = 16
LR = 2.2e-3
WD = 1e-4
POS_WEIGHT = 6.0
LIST_W = 2.0
MIN_FREQ = 3
MAX_VOCAB = 12000
STMT_MIN_FREQ = 6
STMT_MAX_VOCAB = 12000
SEEDS = 1
MAX_PER_PROG = 3
WARMUP_FRAC = 0.06
TOK_DROP = 0.20
SLOT_PERM = 0.0

TOKRE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|<<=|>>=|<=|>=|==|!=|&&|\|\||\+\+|--|->|\+=|-=|\*=|/=|%=|\^=|&=|\|=|::|<<|>>|\S"
)
WORDRE = re.compile(r"[a-z]+")
NUMRE = re.compile(r"\d+")
IDENTRE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

KEYWORDS = set(
    "if else for while do return break continue int long double float char bool void auto const static "
    "struct class public private template typename namespace using include define ifdef endif new delete "
    "this true false null nullptr sizeof unsigned signed short vector string map set pair queue stack deque "
    "printf scanf cout cin endl std def import from print input range len str list dict tuple sorted sum "
    "max min abs lambda self try except raise with as in is not and or none elif pass global yield assert".split()
)

FLAG_PATTERNS = [
    "if", "else", "for", "while", "return", "break", "continue", "def", "class", "int", "long", "double",
    "float", "char", "bool", "void", "vector", "string", "map", "set", "sort", "max", "min", "abs", "printf",
    "scanf", "cout", "cin", "print", "input", "range", "len", "sum", "push_back", "size", "swap", "include",
    "using", "import", "main", "mod", "dp", "ans", "res", "cnt",
]
OP_PATTERNS = [
    "==", "!=", "<=", ">=", "<<", ">>", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=",
    "<", ">", "=", "+", "-", "*", "/", "%", "[", "]", "(", ")", "{", "}", ";", ":", ",", "&", "|", "^", "!", "?", ".",
]
NFEAT = 12 + len(FLAG_PATTERNS) + len(OP_PATTERNS) + 14


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def tok_line(s):
    return TOKRE.findall(s)


def stmt_numbers(txt):
    out = set()
    for m in NUMRE.finditer(txt):
        out.add(m.group(0))
    for m in re.finditer(r"(\d+)\s*\^\s*\{?\s*(\d+)", txt):
        b, e = int(m.group(1)), int(m.group(2))
        if 0 < e < 19 and b < 100:
            out.add(str(b ** e))
    for m in re.finditer(r"(\d+)\s*[eE]\s*(\d+)", txt):
        b, e = int(m.group(1)), int(m.group(2))
        if 0 < e < 19:
            out.add(str(b * 10 ** e))
    return out


def stmt_words(txt):
    return WORDRE.findall(txt.lower())


def read_prog(base, pid):
    code = (base / (pid + ".txt")).read_text(encoding="utf-8", errors="replace")
    stmt = (base / (pid + ".stmt.txt")).read_text(encoding="utf-8", errors="replace")
    return code, stmt


def line_flags(raw, toks, tokset, snums, sidents, lang_cpp, jac_max, dup):
    f = np.zeros(NFEAT, dtype=np.float32)
    st = raw.strip()
    k = 0
    f[k] = 1.0 if lang_cpp else 0.0
    k += 1
    f[k] = min(len(raw), 200) / 80.0
    k += 1
    f[k] = min(len(toks), 40) / 20.0
    k += 1
    ind = 0
    for ch in raw:
        if ch == " ":
            ind += 1
        elif ch == "\t":
            ind += 4
        else:
            break
    f[k] = min(ind, 32) / 8.0
    k += 1
    f[k] = 1.0 if st == "" else 0.0
    k += 1
    f[k] = 1.0 if (st.startswith("//") or (st.startswith("#") and not lang_cpp)) else 0.0
    k += 1
    f[k] = 1.0 if st in ("{", "}", "};", "{}", ")", "),") else 0.0
    k += 1
    f[k] = jac_max
    k += 1
    f[k] = 1.0 if dup else 0.0
    k += 1
    nums = [t for t in toks if t and t[0].isdigit()]
    f[k] = min(len(nums), 6) / 3.0
    k += 1
    inn = sum(1 for t in nums if t in snums)
    f[k] = min(inn, 4) / 2.0
    k += 1
    f[k] = min(len(nums) - inn, 4) / 2.0
    k += 1
    for p in FLAG_PATTERNS:
        f[k] = 1.0 if p in tokset else 0.0
        k += 1
    for p in OP_PATTERNS:
        c = 0
        for t in toks:
            if t == p:
                c += 1
        f[k] = min(c, 4) / 2.0
        k += 1
    idents = [t for t in toks if IDENTRE.match(t) and t not in KEYWORDS]
    f[k] = min(len(idents), 8) / 4.0
    k += 1
    f[k] = min(sum(1 for t in idents if t.lower() in sidents), 4) / 2.0
    k += 1
    f[k] = min(len(set(idents)), 8) / 4.0
    k += 1
    f[k] = 1.0 if len(idents) != len(set(idents)) else 0.0
    k += 1
    f[k] = 1.0 if any(len(t) == 1 for t in idents) else 0.0
    k += 1
    f[k] = 1.0 if ("0" in toks or "1" in toks) else 0.0
    k += 1
    f[k] = 1.0 if st.endswith(",") else 0.0
    k += 1
    f[k] = 1.0 if st.endswith("{") else 0.0
    k += 1
    f[k] = 1.0 if ("=" in toks and "==" not in toks) else 0.0
    k += 1
    f[k] = 1.0 if st.startswith("return") else 0.0
    k += 1
    f[k] = 1.0 if st.startswith("#") else 0.0
    k += 1
    f[k] = 1.0 if st.startswith("using") else 0.0
    k += 1
    f[k] = min(len(raw) - len(raw.lstrip()), 20) / 10.0
    k += 1
    f[k] = 1.0 if ("<" in toks or ">" in toks or "<=" in toks or ">=" in toks) else 0.0
    k += 1
    return f


def build_program(code, stmt, lang, code_vocab, stmt_vocab):
    lines = code.split("\n")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
    n = len(lines)
    lang_cpp = (lang == "C++")
    toks_per = [tok_line(l)[:MAX_TOK] for l in lines]
    sets = [set(t) for t in toks_per]
    jac = np.zeros(n, dtype=np.float32)
    dupf = np.zeros(n, dtype=bool)
    for i in range(n):
        a = sets[i]
        if len(a) < 2:
            continue
        best = 0.0
        for j in range(n):
            if j == i:
                continue
            b = sets[j]
            if not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            v = inter / float(len(a | b))
            if v > best:
                best = v
        jac[i] = best
        if best > 0.85:
            dupf[i] = True
    snums = stmt_numbers(stmt)
    sw = stmt_words(stmt)
    sidents = set(sw)
    slot_map = {}
    tok_ids = np.zeros((n, MAX_TOK), dtype=np.int64)
    slot_ids = np.zeros((n, MAX_TOK), dtype=np.int64)
    feats = np.zeros((n, NFEAT), dtype=np.float32)
    for i, tl in enumerate(toks_per):
        for j, t in enumerate(tl):
            tok_ids[i, j] = code_vocab.get(t, 1)
            if IDENTRE.match(t) and t not in KEYWORDS:
                if t not in slot_map:
                    slot_map[t] = min(len(slot_map) + 1, NSLOT - 1)
                slot_ids[i, j] = slot_map[t]
        feats[i] = line_flags(lines[i], tl, sets[i], snums, sidents, lang_cpp, jac[i], dupf[i])
    pos = np.zeros((n, 4), dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    pos[:, 0] = idx / max(n - 1, 1)
    pos[:, 1] = np.minimum(idx, 60) / 60.0
    pos[:, 2] = np.minimum(n - 1 - idx, 60) / 60.0
    pos[:, 3] = min(n, 128) / 128.0
    feats = np.concatenate([feats, pos], axis=1)
    labelable = np.array([1.0 if l.strip() != "" else 0.0 for l in lines], dtype=np.float32)
    sids = np.zeros(MAX_STMT, dtype=np.int64)
    for j, w in enumerate(sw[:MAX_STMT]):
        sids[j] = stmt_vocab.get(w, 1)
    slen = max(min(len(sw), MAX_STMT), 1)
    mt = int((tok_ids != 0).sum(axis=1).max()) if n else 1
    return {
        "tok": tok_ids, "slot": slot_ids, "feat": feats, "lab": labelable,
        "stmt": sids, "slen": slen, "n": n, "mt": max(mt, 2),
    }


class Net(nn.Module):
    def __init__(self, nvocab, nstmt, nfeat):
        super().__init__()
        self.tok_emb = nn.Embedding(nvocab, EMB, padding_idx=0)
        self.slot_emb = nn.Embedding(NSLOT, SLOT_EMB)
        self.stmt_emb = nn.Embedding(nstmt, 96, padding_idx=0)
        self.conv1 = nn.Conv1d(EMB, CONV1, 3, padding=1)
        self.conv2 = nn.Conv1d(CONV1, CONV2, 3, padding=1)
        self.featp = nn.Sequential(nn.Linear(nfeat, 128), nn.GELU(), nn.Linear(128, 128))
        self.stmtp = nn.Sequential(nn.Linear(96, 96), nn.GELU())
        self.merge = nn.Linear(CONV2 * 2 + 128 + 96, DMODEL)
        self.pos_emb = nn.Embedding(MAX_LINES + 1, DMODEL)
        self.norm_in = nn.LayerNorm(DMODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=DMODEL, nhead=NHEAD, dim_feedforward=FFDIM, dropout=DROP,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, NLAYER)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Sequential(nn.Linear(DMODEL, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, tok, slot, feat, stmt, smask, lmask):
        B, L, T = tok.shape
        e = self.tok_emb(tok) + self.slot_emb(slot)
        e = e.view(B * L, T, EMB).transpose(1, 2)
        h = F.gelu(self.conv1(e))
        h = F.gelu(self.conv2(h))
        tm = (tok.view(B * L, T) != 0).float().unsqueeze(1)
        hmax = (h - 1e4 * (1.0 - tm)).max(dim=2).values
        hsum = (h * tm).sum(dim=2) / tm.sum(dim=2).clamp(min=1.0)
        line = torch.cat([hmax, hsum], dim=1).view(B, L, CONV2 * 2)
        sv = self.stmt_emb(stmt)
        sv = (sv * smask.unsqueeze(2)).sum(1) / smask.sum(1, keepdim=True).clamp(min=1.0)
        sv = self.stmtp(sv).unsqueeze(1).expand(B, L, 96)
        fv = self.featp(feat)
        x = self.merge(torch.cat([line, fv, sv], dim=2))
        ar = torch.arange(L, device=tok.device).unsqueeze(0).expand(B, L)
        x = self.norm_in(x + self.pos_emb(ar))
        x = self.drop(x)
        x = self.enc(x, src_key_padding_mask=(lmask < 0.5))
        return self.head(x).squeeze(-1)


def collate(items, device, drop_tok=0.0, rng=None):
    B = len(items)
    L = max(it["n"] for it in items)
    T = max(it["mt"] for it in items)
    tok = np.zeros((B, L, T), dtype=np.int64)
    slot = np.zeros((B, L, T), dtype=np.int64)
    nf = items[0]["feat"].shape[1]
    feat = np.zeros((B, L, nf), dtype=np.float32)
    lab = np.zeros((B, L), dtype=np.float32)
    tgt = np.zeros((B, L), dtype=np.float32)
    stmt = np.zeros((B, MAX_STMT), dtype=np.int64)
    smask = np.zeros((B, MAX_STMT), dtype=np.float32)
    for i, it in enumerate(items):
        n = it["n"]
        tok[i, :n] = it["tok"][:, :T]
        slot[i, :n] = it["slot"][:, :T]
        feat[i, :n] = it["feat"]
        lab[i, :n] = it["lab"]
        if "y" in it:
            tgt[i, :n] = it["y"]
        stmt[i] = it["stmt"]
        smask[i, : it["slen"]] = 1.0
    if drop_tok > 0.0:
        m = (np.random.rand(B, L, T) < drop_tok) & (tok > 0)
        tok = np.where(m, 1, tok)
        if SLOT_PERM > 0.0:
            for i in range(B):
                if np.random.rand() < SLOT_PERM:
                    perm = np.arange(NSLOT)
                    perm[1:] = np.random.permutation(perm[1:])
                    slot[i] = perm[slot[i]]
    tt = lambda a: torch.from_numpy(a).to(device)
    return tt(tok), tt(slot), tt(feat), tt(stmt), tt(smask), tt(lab), tt(tgt)


def make_batches(items, bs, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
        out = []
        mega = 512
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s:s + mega], key=lambda i: items[i]["n"] * items[i]["mt"])
            for b in range(0, len(chunk), bs):
                out.append(chunk[b:b + bs])
        rng.shuffle(out)
        return out
    order = sorted(idx, key=lambda i: items[i]["n"] * items[i]["mt"])
    return [order[b:b + bs] for b in range(0, len(order), bs)]


def run_epochs(model, opt, sched, items, rng, epochs):
    model.train()
    for _ in range(epochs):
        for bidx in make_batches(items, BATCH, True, rng):
            batch = [items[i] for i in bidx]
            tok, slot, feat, stmt, smask, lmask, y = collate(batch, DEVICE, TOK_DROP)
            logit = model(tok, slot, feat, stmt, smask, lmask)
            w = torch.where(y > 0.5, torch.full_like(y, POS_WEIGHT), torch.ones_like(y))
            bce = F.binary_cross_entropy_with_logits(logit, y, weight=w, reduction="none")
            bce = (bce * lmask).sum() / lmask.sum().clamp(min=1.0)
            masked = logit.masked_fill(lmask < 0.5, -1e4)
            logp = F.log_softmax(masked, dim=1)
            denom = y.sum(1).clamp(min=1.0)
            listw = -((logp * y).sum(1) / denom)
            has = (y.sum(1) > 0).float()
            listw = (listw * has).sum() / has.sum().clamp(min=1.0)
            loss = bce + LIST_W * listw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, items):
    model.eval()
    out = [None] * len(items)
    for bidx in make_batches(items, 32, False, None):
        batch = [items[i] for i in bidx]
        tok, slot, feat, stmt, smask, lmask, _ = collate(batch, DEVICE)
        logit = model(tok, slot, feat, stmt, smask, lmask)
        keep = torch.from_numpy(np.stack([np.pad(items[i]["lab"], (0, lmask.shape[1] - items[i]["n"])) for i in bidx]))
        p = torch.log_softmax(logit.masked_fill(keep < 0.5, -1e4), dim=1).numpy()
        for k, i in enumerate(bidx):
            out[i] = p[k, : items[i]["n"]]
    return out


def build_vocabs(codes, stmts):
    c = Counter()
    for t in codes:
        c.update(tok_line(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, f in c.most_common(MAX_VOCAB):
        if f < MIN_FREQ:
            break
        vocab[w] = len(vocab)
    s = Counter()
    for t in stmts:
        s.update(stmt_words(t))
    svocab = {"<pad>": 0, "<unk>": 1}
    for w, f in s.most_common(STMT_MAX_VOCAB):
        if f < STMT_MIN_FREQ:
            break
        svocab[w] = len(svocab)
    return vocab, svocab


def group_key(stmt):
    return hashlib.md5(re.sub(r"\s+", " ", stmt.strip()).encode("utf-8")).hexdigest()


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    set_seed(SEED)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    tr_code, tr_stmt = [], []
    for pid in train["id"]:
        c, s = read_prog(public_dir / "train", pid)
        tr_code.append(c)
        tr_stmt.append(s)
    te_code, te_stmt = [], []
    for pid in test["id"]:
        c, s = read_prog(public_dir / "test", pid)
        te_code.append(c)
        te_stmt.append(s)

    vocab, svocab = build_vocabs(tr_code, tr_stmt)

    tr_items = []
    for i, row in enumerate(train.itertuples()):
        it = build_program(tr_code[i], tr_stmt[i], row.lang, vocab, svocab)
        y = np.zeros(it["n"], dtype=np.float32)
        for x in str(row.label).split():
            k = int(x) - 1
            if 0 <= k < it["n"] and it["lab"][k] > 0.5:
                y[k] = 1.0
        it["y"] = y
        tr_items.append(it)
    te_items = []
    for i, row in enumerate(test.itertuples()):
        te_items.append(build_program(te_code[i], te_stmt[i], row.lang, vocab, svocab))

    keys = [group_key(s) for s in tr_stmt]
    uk = sorted(set(keys))
    rnd = random.Random(SEED)
    rnd.shuffle(uk)
    kfold = {k: i % FOLDS for i, k in enumerate(uk)}
    folds = np.array([kfold[k] for k in keys], dtype=np.int64)

    nfeat = tr_items[0]["feat"].shape[1]
    oof_acc = [np.zeros(it["n"], dtype=np.float64) for it in tr_items]
    test_acc = [np.zeros(it["n"], dtype=np.float64) for it in te_items]

    for f in range(FOLDS):
        tri = [i for i in range(len(tr_items)) if folds[i] != f]
        vai = [i for i in range(len(tr_items)) if folds[i] == f]
        sub = [tr_items[i] for i in tri]
        for sd in range(SEEDS):
            set_seed(SEED + 17 * f + 991 * sd)
            model = Net(len(vocab), len(svocab), nfeat).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            steps = EPOCHS * math.ceil(len(sub) / BATCH)
            warm = max(1, int(steps * WARMUP_FRAC))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))),
            )
            rng = random.Random(SEED + f + 7 * sd)
            run_epochs(model, opt, sched, sub, rng, EPOCHS)
            vp = predict(model, [tr_items[i] for i in vai])
            for k, i in enumerate(vai):
                oof_acc[i] += vp[k]
            tp = predict(model, te_items)
            for i in range(len(te_items)):
                test_acc[i] += tp[i]
            print("fold", f, "seed", sd, "done", flush=True)

    hit = 0
    for i in range(len(tr_items)):
        lb = tr_items[i]["lab"]
        sc = oof_acc[i].copy()
        sc[lb < 0.5] = -1e9
        if len(sc) and tr_items[i]["y"][int(np.argmax(sc))] > 0.5:
            hit += 1
    print("oof top1", round(hit / max(len(tr_items), 1), 4), flush=True)

    rows = []
    for i, row in enumerate(test.itertuples()):
        sc = test_acc[i]
        lb = te_items[i]["lab"]
        cand = [(float(sc[k]), k + 1) for k in range(len(sc)) if lb[k] > 0.5]
        cand.sort(key=lambda z: -z[0])
        sel = [cand[0][1]] if cand else []
        rows.append({"id": row.id, "lines": " ".join(str(x) for x in sorted(sel))})

    submission = pd.DataFrame(rows, columns=["id", "lines"])
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    print("wrote", len(submission), flush=True)


if __name__ == "__main__":
    main()
