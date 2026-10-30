# SC4002 Group Assignment: AI Collaboration Log (Stage 2)

- **Group ID:** G52
- **Dates:** 30 October 2026 to <end date>

## Chat 1
### Owner: Tan Jun Wei, Adison
### AI tool / model used: Claude, Opus 5.5
---

**User**

> hello chat. ive attached the instructions pdf for my nlp group project and my github repo (https://github.com/thisisadison/SC4002). ive finished stage 1 and need help with stage 2.
>
> the task is joint intent classification and slot filling on amazon massive (en-US), with one model for both. its graded on intent accuracy and slot micro f1. rules: max 20M parameters including word vectors, no pretrained models except word vectors, only train on the training split.
>
> my baseline is an lstm (embedding 100, hidden 128) with a tag head and an intent head, cross entropy loss and adam (lr 0.001, 40 epochs). best checkpoint is in the checkpoints folder. results:
> - validation: 76.78% intent, 67.82% slot f1
> - test: 75.39% intent, 62.85% slot f1
>
> can you look at my code and results and tell me whats holding the model back? then suggest improvements one at a time, ranked by how much you think theyd help the test scores, and explain why each one should work. dont suggest stuff i already have. ill test each one before moving on.

**AI**

Read your notebook and training log; now running a few quick data checks to back up the diagnosis.

I went through `notebooks/main.ipynb` and the full Adam training log, and ran a few checks on `en-US.jsonl`. Three things are holding the model back, plus one bug that breaks reproducibility.

## What's holding the model back

**1. The intent head reads padding, not the sentence.** This is the biggest problem. The batches are right-padded and fed to the LSTM without packing. `h[-1]` is the state after the LSTM has also stepped through every `<PAD>` token. The PAD embedding is zero, but the LSTM still updates its state at each pad step because of its biases. In your dev batches, a sentence goes through **10 pad steps on average** before `h[-1]` is read, and **90%** of sentences get at least 5. So the intent head sees a washed-out state that changes with whatever else is in the batch. This fits intent accuracy stalling around 76% from epoch 16 onward. Slot tags are mostly unaffected, because in a left-to-right LSTM the padding comes after the real words.

**2. Each tag only sees the words before it.** With a one-directional LSTM, the tag for "X" in "play X by Y" can't see "by". Deciding where a slot starts and ends (B vs I) often depends on the words after it.

**3. Rare and unseen words get poor representations.** I measured this:

| | Dev | Test |
|---|---|---|
| Unseen words inside gold slot spans | 8.1% | **9.0%** |
| Unseen words outside slots | 1.9% | 2.0% |
| Sentences with at least one unseen word | 17.9% | 18.9% |

Unseen words are about 4× more common inside slots, and test has more of them than dev. That's a likely reason test slot F1 (62.85%) falls 5 points below validation (67.82%). It's also worse than it looks:
- **The `<UNK>` vector never gets trained.** Every training word is in the vocabulary, so `<UNK>` never appears during training and keeps its random starting values. Every unseen test word is mapped to that random vector.
- **2,537 of your 5,302 words appear only once** in training, so their learned vectors are close to noise.

**4. Heavy overfitting with no regularisation.** Validation loss is lowest at epoch 16 (1.54) and rises to 1.89 by epoch 40, while training loss falls to 0.17. There's no dropout anywhere.

**Reproducibility bug (separate from scores):** cell 37 uses `LEARNING_RATE`, which isn't defined in cell 1. If the graders do Restart & Run All, it will stop with a `NameError`. Cell 49 also evaluates `model` (the epoch-40 weights) instead of `eval_model`, so it prints numbers that don't match the checkpoint. Cell 50 is correct, so I'd delete cell 49.

## Improvements, ranked by expected effect on test scores

1. **Pack the padded batches** so the LSTM stops at each sentence's real last word. Fixes problem 1. Expect a large intent gain.
2. **Make the LSTM bidirectional**, so each tag sees the words on both sides. Fixes problem 2. Expect the largest slot F1 gain from an architecture change. This needs #1 first, or the backward direction would start on padding.
3. **Use pretrained GloVe 300d word vectors**, including vectors for words outside the training vocabulary. Fixes problem 3. Expect gains on both metrics and a smaller gap between validation and test. #2 and #3 are close; on test, GloVe could come out ahead.
4. **Add dropout** on the embeddings and LSTM outputs (around 0.3–0.5). Fixes problem 4.
5. **Randomly replace training words with `<UNK>`** (word dropout, often only for rare words). This finally trains the `<UNK>` vector.
6. **Add a CRF layer, or at least constrained decoding,** so the model can't predict invalid tag sequences such as an I- tag with no B- before it. These currently turn into dropped or broken slots.
7. **Add character-level word features** (a small character CNN) so unseen names and numbers still get a meaningful representation. Smaller gain once GloVe is in.
8. **Pool over all LSTM outputs for intent** (max-pooling or attention) instead of using only the final state.

After that it's tuning: hidden size, 2 layers, learning-rate schedule, gradient clipping.

## Start with #1: pack the padded sequences

Change only `forward`:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def forward(self, sentence):
    lengths = (sentence != 0).sum(dim=1).cpu()   # PAD=0; real tokens are always >= 1
    embeds = self.word_embeddings(sentence)
    packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
    packed_out, (h, c) = self.lstm(packed)
    lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                      total_length=sentence.size(1))  # keep shape aligned with tags
    tag_space = self.hidden2tag(lstm_out)
    intent_space = self.hidden2intent(h[-1])     # now the state at each sentence's real last word
    return tag_space, intent_space
```

**Why it should work:** with packed input, the LSTM stops at each sentence's true length. `h[-1]` then summarises the actual words, and the result no longer depends on what else is in the batch.

**How to test it:** keep everything else the same (seed, Adam at 0.001, 40 epochs, same checkpoint rule) and compare against your baseline.

**What the result should look like if I'm right:** intent accuracy goes up clearly, and slot F1 barely changes, since tags were already unaffected by padding. If slot F1 also jumps or intent doesn't move, the diagnosis is off and we should look again. This prediction is also useful for the report section on how you checked the AI's advice.

Send me the validation and test numbers when you have them, and we'll move on to #2.

---

**User**

> <your next message>

**AI**

<the AI's next reply>

---
