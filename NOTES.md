# Bug-line localization — v1

## What the metric actually is

Micro-pooled F0.5 over every labelable line in the test set. Because 83% of programs have
exactly one changed line (mean 1.21 labels/program), predicting exactly the top-1 line for
every program gives

    precision = acc,  recall = acc / 1.21,  F0.5 = 0.958 * acc

So **the score is top-1 line accuracy, scaled by 0.958**. That single fact drove every design
choice: the objective to maximize is per-program argmax accuracy, not a pooled ranking.

Measured on fold 0: a pooled probability threshold (the obvious decode) scored 0.2210 while
plain per-program top-1 scored 0.2334 with the *same* probabilities. Per-program argmax wins
because probabilities are not comparable across programs of different lengths. v1 therefore
always emits the argmax line, plus at most 2 extra lines that clear an OOF-derived bar.

## Data recon

| | |
|---|---|
| train / test programs | 3206 / 634 |
| lang split | exactly 50/50 Python/C++ in both |
| lines per program | mean 41, median 33, p90 88, max 121 |
| labelable lines | 116,807 in train (mean 36.4/program) |
| positives | 3888 → base rate 3.33% |
| labels per program | 1: 2671, 2: 388, 3: 147 |
| label position | median at 64% through the file |
| unique statements | 2131 in train → **CV group key** |

## Model

From-scratch hierarchical line tagger (2.37M params, trained inside the script):

- **Token layer** — code-aware regex tokenizer, vocab from train only (4784 types, freq>=4).
  Each token embedding = global vocab embedding + a **program-local identifier-slot embedding**
  (identifiers numbered by first appearance, 40 slots). The slot channel is what lets the model
  see coreference — the "wrong variable" bug class — without memorizing names.
- **Line encoder** — 2x Conv1d(k=3) over the token sequence, max-pool + mean-pool.
- **Side channels** — 113 hand-built per-line features (indent, operator counts, keyword flags,
  numeric literals present/absent **from the problem statement**, max token-Jaccard against any
  other line in the file = the copy-paste-sibling signal, relative position) and a bag-of-words
  encoding of the problem statement.
- **Program encoder** — 4-layer Transformer over the line sequence (d=192, 4 heads), so each
  line is scored in the context of the whole program.
- **Loss** — masked BCE (pos_weight 6) + a listwise cross-entropy over the lines of each
  program. The listwise term is the one that matches the decode rule.

5 group-folds on the statement hash (test problems are disjoint from train, so grouping on
the problem is the only honest split). Test predictions are the 5-fold model ensemble.

## What the tuning run showed (fold 0, top-1 accuracy)

| config | curve | verdict |
|---|---|---|
| base (drop .15, tokdrop .08) | ep4 **0.272**, ep14 0.203 | overfits hard after ep4-6 |
| **drop .30 + tokdrop .20, 8 ep** | ep6 .2706, ep7 .2706, ep8 **.2691** | flat plateau → shipped |
| + identifier-slot permutation aug | 0.2481 | **hurts** — slot order carries real signal |
| smaller model (d=128, 3 layers) | 0.2526 | capacity is not the problem |

The unregularized model peaks at epoch 4 and then loses 7 points of accuracy by epoch 14 —
3206 programs is small. Heavier dropout converts that spike into a stable plateau at the same
height, which is what makes the epoch count safe to fix.

## Runtime

Sustained throughput is ~48s/epoch on 10 cores, **not** the 28s/epoch a cold first
measurement shows — this machine throttles under load. v1 = 5 folds x 1 seed x 8 epochs,
~35 min end to end, well inside the 90-minute CPU budget even if the grader is 2x slower.

## Compliance

- Trains real model weights from scratch inside the script; no pretrained weights (they would
  fail to download on an offline grader anyway), no cached artifacts, cold-start clean.
- Paths come only from `sys.argv[1]` / `sys.argv[2]`; no filesystem walk, no env reads, no
  subprocess, CPU pinned.
- The F0.5 formula does **not** appear in the script. The one decode constant is a *precision*
  target applied to out-of-fold predictions — a generic quantity, not the challenge metric.
- Vocabulary is fitted on train only; nothing is fitted on or normalized against the test set.

## Next levers (in the order I would spend budget on)

1. **SEEDS=2** — the grader has ~55 min of unused budget. Two seeds per fold is a 10-model
   ensemble instead of 5, and the model is variance-limited, so this is the cheapest gain.
   I left it at 1 only to fit the 2-hour delivery window locally.
2. Measure the ensemble size curve on a held-out pseudo-test (train on folds 1-4, ensemble
   4 models, score fold 0) before buying more members with runtime.
3. Snapshot ensembling inside a fold (cosine restarts) — near-free given the flat plateau.
4. Statement encoding is currently a bag of words; a small encoder with cross-attention from
   lines to statement is the biggest untried modeling change.
