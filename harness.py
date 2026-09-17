import sys, time, math, random, argparse
import numpy as np
import pandas as pd
import torch
import solution as S

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--run_folds", type=int, default=1)
ap.add_argument("--epochs", type=int, default=7)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--tag", type=str, default="h")
ap.add_argument("--evalevery", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
S.EPOCHS = a.epochs
D = "."

t0 = time.time()
train = pd.read_csv(D + "/train.csv")
tr_code, tr_stmt = [], []
from pathlib import Path
for pid in train["id"]:
    c, s = S.read_prog(Path(D) / "train", pid)
    tr_code.append(c); tr_stmt.append(s)
vocab, svocab = S.build_vocabs(tr_code, tr_stmt)
print("vocab", len(vocab), len(svocab), "nfeat", S.NFEAT + 4)
items = []
for i, row in enumerate(train.itertuples()):
    it = S.build_program(tr_code[i], tr_stmt[i], row.lang, vocab, svocab)
    y = np.zeros(it["n"], dtype=np.float32)
    for x in str(row.label).split():
        k = int(x) - 1
        if 0 <= k < it["n"] and it["lab"][k] > 0.5:
            y[k] = 1.0
    it["y"] = y
    items.append(it)
print("prep %.1fs" % (time.time() - t0))

keys = [S.group_key(s) for s in tr_stmt]
uk = sorted(set(keys)); rnd = random.Random(S.SEED); rnd.shuffle(uk)
kf = {k: i % a.folds for i, k in enumerate(uk)}
folds = np.array([kf[k] for k in keys])

nfeat = items[0]["feat"].shape[1]
oof = [None] * len(items)
done = []
for f in range(a.run_folds):
    tf = time.time()
    S.set_seed(S.SEED + 17 * f)
    tri = [i for i in range(len(items)) if folds[i] != f]
    vai = [i for i in range(len(items)) if folds[i] == f]
    sub = [items[i] for i in tri]
    model = S.Net(len(vocab), len(svocab), nfeat)
    if f == 0:
        print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=S.LR, weight_decay=S.WD)
    steps = S.EPOCHS * math.ceil(len(sub) / S.BATCH)
    warm = max(1, int(steps * S.WARMUP_FRAC))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = random.Random(S.SEED + f)
    for ep in range(S.EPOCHS):
        S.run_epochs(model, opt, sched, sub, rng, 1)
        if (ep + 1) % a.evalevery == 0 or ep == S.EPOCHS - 1:
            vq = S.predict(model, [items[i] for i in vai])
            hit = 0
            for k, i in enumerate(vai):
                lb = items[i]["lab"]; pp = vq[k].copy(); pp[lb < 0.5] = -1
                if items[i]["y"][int(np.argmax(pp))] > 0.5:
                    hit += 1
            print("   ep %d top1 %.4f  (%.1fs)" % (ep + 1, hit / len(vai), time.time() - tf), flush=True)
    vp = S.predict(model, [items[i] for i in vai])
    for k, i in enumerate(vai):
        oof[i] = vp[k]
    done += vai
    print("fold %d  %.1fs  (%.1fs/epoch)" % (f, time.time() - tf, (time.time() - tf) / S.EPOCHS), flush=True)


def f05(tp, npred, npos):
    if npred == 0 or tp == 0:
        return 0.0
    p = tp / npred; r = tp / npos
    return 1.25 * p * r / (0.25 * p + r)


sc = []
npos = 0
for i in done:
    p = oof[i]; y = items[i]["y"]; lb = items[i]["lab"]
    npos += int(y.sum())
    order = np.argsort(-p)
    rank = 0
    for k in order:
        if lb[k] < 0.5:
            continue
        rank += 1
        sc.append((float(p[k]), float(y[k]), rank))
sc.sort(key=lambda z: -z[0])
N = len(done)
print("progs %d  labelable %d  pos %d" % (N, len(sc), npos))

best = (0, 0, 0)
cum = 0.0
curve = []
for k, (pv, yv, rk) in enumerate(sc, 1):
    cum += yv
    v = f05(cum, k, npos)
    curve.append((k, pv, cum / k, v))
    if v > best[0]:
        best = (v, k, pv)
print("BEST F0.5 %.4f at k=%d (%.2f per prog) thr=%.4f cumprec=%.3f" % (best[0], best[1], best[1] / N, best[2], next(c[2] for c in curve if c[0] == best[1])))
# top1-only baseline
tp1 = sum(1 for pv, yv, rk in sc if rk == 1 and yv > 0.5)
print("top1-only: acc %.4f  F0.5 %.4f" % (tp1 / N, f05(tp1, N, npos)))
tp2 = sum(1 for pv, yv, rk in sc if rk <= 2 and yv > 0.5)
print("top2-only: F0.5 %.4f" % f05(tp2, 2 * N, npos))
for C in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    cum = 0.0; kk = 0; tt = 1.0
    for k, (pv, yv, rk) in enumerate(sc, 1):
        cum += yv
        if cum / k >= C:
            kk = k; tt = pv
    if kk:
        cum2 = sum(1 for pv, yv, rk in sc[:kk] if yv > 0.5)
        print("  prec_target %.2f -> k=%d (%.2f/prog) thr=%.4f F0.5=%.4f" % (C, kk, kk / N, tt, f05(cum2, kk, npos)))
np.save("oof_%s.npy" % a.tag, np.array([1]))
import pickle
pickle.dump({"oof": oof, "done": done, "folds": folds}, open("oof_%s.pkl" % a.tag, "wb"))
print("total %.1fs" % (time.time() - t0))
