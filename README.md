# SC4002 — Joint Intent Classification & Slot Filling

NLP group assignment (AY26-27 Sem 1). For each English assistant utterance, one
joint model predicts the **intent** (one of 60 labels, e.g. `alarm_set`) and
extracts its **slots** (`[slot type: slot span]` pairs, e.g. `[time: five am]`)
using the [MASSIVE 1.0](https://github.com/alexa/massive) en-US dataset.

Everything lives in `notebooks/main.ipynb`. The rest of this README walks through
what that notebook does, top to bottom, so you can follow the flow before opening it.

## The two stages

- **Stage 1 — the baseline.** A plain unidirectional LSTM, built without any AI
  help. This is the starting point we try to beat.
- **Stage 2 — the improvements.** Six experiments done in collaboration with AI.
  Each one looks at where the previous model fell short, makes a single targeted
  change, and checks whether it actually helped. The winner becomes the final system.

## How the notebook flows

**1. Exploring the data.**
Load MASSIVE and split it into train / validation / test using the dataset's own
`partition` field (and assert the official sizes: 11,514 / 2,033 / 2,974). Then look
around — how intents are distributed, how long utterances are, how many words in the
dev/test sets are never seen in training, and how often each slot type appears. These
checks motivate later decisions (for example, lots of unseen words is why GloVe is
tried in Stage 2).

**2. Turning text into model inputs.**
Build a vocabulary from the training words and map each word to an integer id. Slots
are rewritten in **BIO tagging** — every word gets a tag like `B-time` (beginning of a
time slot), `I-time` (inside/continuation), or `O` (not a slot). So one utterance
produces three aligned things: word ids, a BIO tag per word, and one intent label.
Sentences are then padded and batched so they can go through the model together.

**3. Scoring (shared by every experiment).**
Two metrics, defined once and reused everywhere so comparisons are fair:
- **Intent accuracy** — did we predict the right intent?
- **Slot micro-F1** — did we extract the exact right `[slot type: span]` pairs? A
  match only counts if both the type *and* the full span are correct.

**4. Stage 1: the baseline model.**
A single network with two heads sharing one LSTM encoder: one head predicts the
intent for the whole sentence, the other predicts a BIO tag for each word. It is
trained with both an SGD and an Adam run; Adam wins, and the best epoch's checkpoint
is saved. Loading that checkpoint and evaluating on the test set gives the baseline
numbers below.

**5. Stage 2: six experiments.**
Each builds on the previous best and changes one thing:

| # | Change | Why |
|---|---|---|
| 1 | **Packed sequences** | Stop the LSTM reading padding, so the sentence summary isn't blurred. |
| 2 | **Bi-directional LSTM** | Let each word see context *after* it too, which helps slot boundaries. |
| 3 | **Dropout** | Exp 2 overfit quickly; dropout forces the model to generalise. |
| 4 | **Gradient clipping + cosine LR schedule** | Try to stabilise late-training wobble (this one didn't help and was dropped). |
| 5 | **GloVe embeddings** | Start words from meaningful pretrained vectors instead of random ones. |
| 6 | **CRF output layer** | Score the whole tag sequence at once so illegal BIO patterns are ruled out. |

Each experiment keeps its per-epoch training/validation curves and saves the best
checkpoint, so the choices are backed by logged evidence rather than guesswork.

**6. Final evaluation, error analysis, and summary.**
The best Stage 2 checkpoint (Exp 6, the CRF model) is loaded and evaluated on the test
set. A short error analysis then prints real failure cases and explains the likely
causes (e.g. ambiguous words like "pink", or implicit commands with no keywords).
The notebook ends with a Summary comparing every experiment.

## Results

Final test-set performance:

| System | Intent accuracy | Slot micro-F1 |
|---|---|---|
| Stage 1 baseline (unidirectional LSTM) | 75.39% | 62.85% |
| Stage 2 final (BiLSTM + dropout + GloVe + CRF) | 84.70% | 75.85% |

The whole system is 820,288 parameters — well under the 20M cap. Per-experiment
validation and test numbers, with analysis, are in the notebook's Summary section.

## Files

```
notebooks/main.ipynb   Everything described above, with its logs and outputs.
ai_collaboration.md    The AI chat history behind the Stage 2 experiments.
data/en-US.jsonl       The MASSIVE en-US dataset.
```

MASSIVE 1.0 is used under its original license — see `data/LICENSE`, `data/NOTICE.md`,
and `data/CITATION.md`.
