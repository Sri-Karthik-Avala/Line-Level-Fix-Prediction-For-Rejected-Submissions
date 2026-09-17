# Line-Level Fix Prediction For Rejected Submissions

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | NLP |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-09-06 |

## Problem statement

### Background

Each item is the raw source code of a program that a person submitted to an automated programming judge and that **the judge rejected** — the program compiles and runs to completion, but it produces a wrong answer, runs too slowly (time-limit exceeded), or hits a runtime error. You are also given the natural-language statement of the problem the program was trying to solve.

The program contains a small logic mistake. The author later fixed it, changing only a few lines, and the fixed version was accepted. Your job is to find the mistake **statically**: you do not get the failing input, an execution trace, or a compiler error (the program already compiles and runs). The wrong line looks almost exactly like its correct neighbours — a flipped comparison, an off-by-one bound, the wrong variable, a `+`/`-` swap — so locating it requires reasoning about the whole program in the context of the problem.

### Task

For each test program, predict the set of **1-indexed line numbers of that program** that were modified or deleted to fix the bug. A line is *labelable* if it is non-blank; only labelable lines can be part of the answer, and predictions on blank lines are ignored.

### Files

The dataset contains the following files.

- `train.csv` — one row per training program. Columns:
   - `id` (string): unique identifier of the program; also the basename of its code and statement files.
   - `lang` (string): the programming language, either `Python` or `C++`.
   - `label` (string): space-separated 1-indexed line numbers of `train/<id>.txt` that were changed to fix the bug (the supervision signal). May contain one to three numbers.
- `train/<id>.txt` — the buggy source code of training program `<id>` (UTF-8 text, one program per file; lines are 1-indexed top to bottom).
- `train/<id>.stmt.txt` — the natural-language problem statement for training program `<id>`.
- `test.csv` — one row per test program. Columns:
   - `id` (string): unique identifier of the test program.
   - `lang` (string): the programming language, either `Python` or `C++`.
- `test/<id>.txt` — the buggy source code of test program `<id>` (labels withheld).
- `test/<id>.stmt.txt` — the natural-language problem statement for test program `<id>`.
- `sample_submission.csv` — a correctly-formatted example submission (see below).

The training and test programs come from **disjoint problems**: no problem in the test set appears in training, so the task rewards generalizing bug-localization reasoning rather than memorizing problem-specific code.

### Submission format

Produce a CSV with a header row and exactly two columns:

- `id` (string): the test program id. Every id in `test.csv` must appear exactly once.
- `lines` (string): space-separated 1-indexed line numbers you predict were changed. May be empty (which scores zero for that program).

Example (matching `sample_submission.csv`):

```
id,lines
s0f1e2d3c4b5a6f7,4
s9a8b7c6d5e4f3a2,12 13
s1c2d3e4f5a6b7c8,
```

### Evaluation

Submissions are scored with a **precision-weighted micro F-score (F₀.₅)** pooled over all labelable lines of the whole test set. With `TP` = predicted lines that were truly changed, `FP` = predicted lines that were not, and `FN` = truly-changed lines that were missed:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F0.5      = 1.25 * precision * recall / (0.25 * precision + recall)
```

Predicted lines outside a program's labelable (non-blank) set are discarded before scoring. Because the score weights precision over recall (β = 0.5), **over-flagging is penalized** — blanketing a file with guesses collapses precision and the score. An empty submission scores at the metric floor.

### Rules

- CPU only; no GPU; no internet at run time.

### Notes

Labels are derived from real human fixes and are therefore mildly noisy; a small number of changed lines may reflect incidental edits. Bug difficulty ranges widely — from a single wrong operator to a multi-line rework — which is reflected in the spread of achievable scores.
