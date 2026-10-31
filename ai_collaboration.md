# SC4002 Group Assignment: AI Collaboration Log (Stage 2)

- **Group ID:** G52
- **Dates:** 30 October 2026 to <end date>

## Chat 1: Tan Jun Wei, Adison
- **AI tool / model used:** Claude, Opus 5.5
  
---

**User**

> hello.
>
> ive attached the instructions pdf for my nlp group project and main.ipynb. i need to start on stage 2. please only explain concepts and suggest improvements, dont write code for me, ill implement everything myself.
>
> task is joint intent classification and slot filling on amazon massive (en-US), one model for both. graded on intent accuracy and slot micro f1. rules: max 20M params including word vectors, pretrained word embeddings ok but no pretrained models, only train on the training split.
>
> baseline (stage 1): lstm (embedding 100, hidden 128), tag head + intent head, cross entropy, adam lr 0.001, 40 epochs. val 76.78% intent / 67.82% slot f1, test 75.39% / 62.85%.
>
> what do you think is holding the model back, and what should i try? explain why each idea should help.

**AI**

Reading through your notebook now to see how the baseline is built before I suggest anything.

I read through the whole notebook and the instructions. Here's what I think is holding the baseline back, most important first, and why each fix should help. I haven't written any code.

### What's holding it back

**1. The intent head reads the hidden state after the padding (likely a bug)**

`h[-1]` comes from an LSTM run over the padded batch, and you don't pack it. So for a 3-word utterance in a batch padded to 19, the intent head sees the state after 16 extra steps of zero-vector input. That has two effects:

* The sentence summary gets washed out by steps that carry no information. Short utterances suffer most.
* The prediction depends on batch composition. The same sentence gets a different representation depending on the longest sentence in its batch, and with shuffling that changes every epoch.

Fix (concept): make the LSTM stop at each sequence's real length (packed sequences), or pick out the hidden state at the last real token. This matters even more once you go bidirectional: without packing, the backward direction would start on the padding.

**2. Unidirectional LSTM: each tag sees only the words to its left**

Slot boundaries often depend on what comes next. In "five am", the word after "five" is what makes it `B-time`. In "play X by Y", the "by" tells you X was a song. When the model tags a word, it can't see any of the words after it yet. A BiLSTM gives every token context from both sides. This is the standard fix and probably your biggest single gain on slot F1.

**3. Heavy overfitting and no regularisation**

Your log shows it: train loss drops to 0.17, but val loss bottoms out at epoch 16 (1.54) and climbs to 1.89 by epoch 40. Intent accuracy flattens at about 76–77% from epoch 18. Training only on 11.5k sentences, with no dropout and no weight decay, the model memorises.

* Dropout on embeddings and on the LSTM outputs stops the model leaning on a few features.
* Weight decay, gradient clipping (LSTMs can have occasional gradient spikes) and a learning-rate schedule (e.g. halve the LR when val score stalls) help it settle instead of drifting.
* Val loss rising while F1 still rises means the model is becoming overconfident on its mistakes. Label smoothing on the intent loss is a cheap thing to try for that.

**4. Unknown words, and the `<UNK>` vector is never trained**

20.6% of the distinct words in val and 24.5% in test never appear in train. Every training word is in your vocab, so the `<UNK>` embedding is never used in training. At test time, every unseen word maps to a random untrained vector, and many of those words are exactly the slot values (names, places, songs).

This also fits the gap you see: slot F1 drops ~5 points from val to test while intent only drops ~1.4, and test has more OOV words. Slots depend on the exact words much more than intent does.

Three fixes, which can be combined:

* Pretrained word vectors (e.g. GloVe): words never seen in training still land near similar words ("margaret" near other names), so the model can generalise to them.
* Word dropout / UNK replacement: during training, randomly swap some tokens (especially rare ones) for `<UNK>`. This trains the `<UNK>` vector and teaches the model to tag from context when it doesn't know the word.
* Character-level encoder (small char-CNN or char-BiLSTM, concatenated to the word embedding): it learns sub-word patterns like "-ing", "-s" and "o'clock" from the training data alone, and gives unseen words a meaningful vector. It adds very few parameters.

**Key constraint for pretrained embeddings: the 20M cap**

Full GloVe 6B-300d is 400k words × 300 = 120M parameters, way over. Frozen vectors still count toward the cap. At 300d you can afford roughly 50–60k words and still leave room for the model.

How you choose that subset matters for the rules:

* Defensible: your train vocab plus the top-N most frequent GloVe words. GloVe's file is already in frequency order, so this doesn't depend on your data.
* Leakage: picking GloVe words because they appear in dev/test. Don't do this. If you're unsure, check with the TA and say what you did in the report.

Freeze or fine-tune? If you fine-tune, the training words drift away from the untouched GloVe words, and the advantage for unseen words shrinks. Try frozen vs. fine-tuned (or fine-tuned with a lower LR) and compare on val.

### Improvements to the architecture

**5. CRF layer (or constrained decoding) on the tag head**

Right now each tag is an independent argmax, so the model can output impossible sequences like `O → I-time` or `B-date → I-time`. Your `extract_slots` silently drops those, so each one costs a false negative.

* A CRF learns tag-to-tag transition scores and decodes the best whole sequence (Viterbi). Slot F1 is exact-span match, and one wrong token kills the whole slot, so deciding the sequence as a whole helps directly.
* Cheaper version: keep softmax training and only apply Viterbi with hard BIO constraints at inference.

**6. Better intent pooling**

Even after fixing #1, a single final state is a bottleneck. With a BiLSTM, max-pooling or attention pooling over all token states lets the intent head focus on the deciding words ("alarm", "weather") wherever they are in the sentence.

**7. Let the two tasks inform each other**

Right now they only share the encoder. Knowing the intent narrows down which slots are possible: `song_name` is unlikely under `alarm_set`. One option is to concatenate the pooled sentence vector (or a predicted-intent embedding) to each token before the tag head. Slot-gated models are the published version of this idea. Treat it as a later experiment.

**8. Capacity**

You're at 0.67M of a 20M budget. A 2-layer BiLSTM with hidden size ~256 per direction, plus dropout, is reasonable. Add capacity after regularisation is in place, or it will just overfit faster.

**9. Loss balance**

Tag loss is averaged over tokens and intent loss over sentences, so they're on different scales. A weight on one of them (e.g. `L = L_tag + λ·L_intent`) is worth a small sweep on val.

### How to run the experiments (this also helps the critical-thinking marks)

* Change one thing at a time, in roughly this order: fix padding → BiLSTM + dropout → GloVe → word dropout / char encoder → CRF → pooling / interaction → tuning. Log val after each step. This gives you an ablation table for the report and shows which ideas actually helped, including the ones that didn't.
* Run 2–3 seeds for the final configs. Differences under about 0.5 points may just be noise.
* For error analysis: compute OOV at the token level and inside gold slot spans (more useful than distinct-word counts), plus per-slot-type F1 and intent confusions. Classes with few training examples (e.g. `movie_type` has only 10 slot instances) will likely stay weak, which is a good limitations point.

### Small notebook issues

* Cell 37 uses `LEARNING_RATE`, which isn't defined in the constants cell (you have `LEARNING_RATE_SGD` and `LEARNING_RATE_ADAM`). A fresh "restart & run all" will crash with a NameError, and that hurts the reproducibility marks.
* The `'\['` pattern in cell 11 triggers a SyntaxWarning. A raw string fixes it.

A from-scratch BiLSTM-CRF with GloVe and char features should land clearly above your baseline. My rough guess is mid-80s intent and low-to-mid 70s slot F1, but treat that as a ballpark, not a promise. This conversation can also go into your `ai_collaboration` file, and you can mark which of these suggestions you accept, change or reject.

---

**User**

> ok i did #1 first. packed the sequences so the lstm stops at each sentences real length, everything else same as baseline (seed, adam 0.001, 40 epochs).
>
> best epoch 26: intent 81.36%, slot f1 66.49%, combined 73.93%
> baseline was epoch 33: intent 76.78%, slot f1 67.82%, combined 72.30%
>
> intent went up about 4.6 which matches what you said about the padding washing out the summary. intent also learns way faster now, 35% after epoch 1 vs stuck at 6% for the baseline.
>
> but slot f1 actually dropped a bit (about 1.3). is this expected behaviour?
>
> can i try to add bilstm next with packed sequence?

**AI**

Yes, a small slot F1 drop is expected here, and it's probably not a real regression. And yes, go ahead with the BiLSTM.

### Why slot F1 moved

Packing shouldn't change the tag outputs at all for a left-to-right LSTM. The padding always comes after the real tokens. So when the LSTM produces the state at real token t, it has only seen tokens 1..t, with or without padding. With the same weights, the tag scores at real positions are identical in both versions. The bug only ever affected the intent head, which read the state at the end.

So the 1.3 drop comes from somewhere else. Most likely causes, in order:

1. You picked an earlier checkpoint. You select on the combined score, and intent now peaks earlier, so the best epoch moved from 33 to 26. In your baseline log, slot F1 was still rising late: 67.00 at epoch 26, 67.82 at 33, 68.04 at 39. You're reading the new model's slot F1 at a point where slots hadn't finished improving.
2. The two tasks now compete more for the same encoder. The intent loss now produces a strong, useful gradient instead of noise. Both tasks pull on the same 128 hidden units, which can cost the tagger slightly. This is a real but probably minor effect.
3. Run-to-run noise. Same seed, but the training path changes once the gradients change. A ~1-point swing in slot F1 between runs is normal on 2k validation sentences.

How to check (useful for the "how we verified the AI's advice" part of the report):

* Look at the best slot F1 across all 40 epochs in the new run, not just at the chosen checkpoint. If it's around 67–68, the drop is just checkpoint selection.
* Optional sanity test: run one batch through the same weights with and without packing, and confirm the tag scores at real positions match. That shows directly that packing only fixed the intent path.
* If you have time, run 2–3 seeds of both versions before claiming anything about slots.

Your epoch-1 numbers (35% vs stuck at 6%) are good evidence for the report. They show the intent head was getting a garbage signal before.

### BiLSTM next

Packing now does matter for the tags: without it, the backward direction would start on the padding and carry that noise into every real token's state. Things to watch:

* Head sizes double. Each token's output is forward and backward joined, so both heads take 2 × hidden as input.
* Intent input: use the final state from both directions, joined. Forward's final state sits at the last real token. Backward's final state sits at the first token, after reading the whole sentence right to left. With packing, both are correct. For one layer they're the last two entries of `h`; for more layers, take the last layer's pair. Check the layout in the PyTorch docs rather than guessing.
* Comparing fairly: keeping hidden at 128 per direction roughly doubles the LSTM parameters. So part of any gain could come from capacity rather than seeing both sides. If you want a clean ablation, also try 64 per direction to roughly match the parameter count. Even one extra run makes this a stronger point in the report.
* Expect more overfitting. It will likely fit train faster and val loss may turn up earlier. That's the cue for dropout as the next step. Don't add it in this same run, so you can attribute the gain.

I'd expect the BiLSTM to give a clear slot F1 gain, since right-hand context is exactly what was missing for boundaries like "five am". Intent may improve a bit too.

---

**User**

> ok checked what you said. the best slot f1 across all 40 epochs in the packed run was 66.93 at epoch 28, so not 67-68.
>
> then i did the bilstm, packed, hidden 128 per direction, same settings otherwise.
>
> best epoch 28: intent 81.16%, slot f1 70.21%, combined 75.69%
> packed lstm was epoch 26: intent 81.36%, slot f1 66.49%, combined 73.93%
>
> slot f1 went up about 3.7 like you said. intent stayed flat. i think thats because packing already fixed the main intent issue and most sentences are short, so the forward state already covers the whole sentence. does that make sense?
>
> overfitting kicks in earlier like you expected. train loss goes almost to 0 (0.007) and val loss is lowest at epoch 11 vs 16 for the baseline, then goes up. theres also a spike in train loss at epoch 33. val loss is still lower than the baseline overall tho.
>
> should i consider dropout next?

**AI**

Yes, dropout is the right next step. On your intent reasoning: the conclusion is right, but the explanation needs a small correction.

### First, my packed-run prediction was off

I said the best slot F1 across epochs would be around 67–68, and it was 66.93. So checkpoint selection explains only about 0.4 of the 1.3 drop (66.49 → 66.93). The other ~1 point is either the tasks competing for the shared encoder or run-to-run noise. One run per version can't tell those apart; only multiple seeds can. For the report, I'd call it "a small slot F1 drop, partly from checkpoint selection, rest unresolved". That's more accurate than claiming it was explained.

### Why intent stayed flat

Packing fixed the intent issue, and that's the main reason. But the deeper point is about where each task reads from:

* Intent reads a summary of the whole sentence. After packing, the forward final state has already read every word, whatever the sentence length. Adding the backward direction gives a second summary of the same words in reverse order. That adds little new information.
* Tags read every word's own state. In a forward-only LSTM, the word "five" has never seen "am". This is where both directions help, which is why the gain went to slots.

Short sentences help a bit too, since the LSTM forgets little over ~7 words. But that's secondary: even long sentences would be fully read by the final state. So intent's remaining errors are elsewhere: overfitting, unknown words, and squeezing the sentence into one vector (the pooling idea from before).

### About what you're seeing

* Train loss ≈ 0.007 and val loss lowest at epoch 11: the bigger model memorises faster. Val loss rising while F1 keeps improving until epoch 28 means the model is still getting more answers right but becoming overconfident on the ones it gets wrong.
* Train-loss spike at epoch 33: this is a known effect of training with Adam once the loss is close to zero. Adam scales each step by the recent size of the gradients. When the gradients have been tiny for a while, one occasional large gradient (LSTMs produce these) gets a very large step. Gradient clipping and a decaying learning rate are the usual fixes. They're about stability, not regularisation, so they're a separate experiment from dropout.

### Dropout: what to watch for

* Where: on the embeddings (before the LSTM) and on the BiLSTM outputs (before both heads). Note: the `dropout` argument of `nn.LSTM` only applies between stacked layers, so with 1 layer it does nothing (PyTorch warns about this). You need separate dropout layers.
* Rate: try 2–3 values, e.g. 0.2 / 0.3 / 0.5, and pick on val. Slot tagging can suffer if embedding dropout is too high, because it relies on the exact word.
* What should change if it's working: train loss stays higher, the lowest val loss arrives later, and the train–val gap shrinks. Your best epoch will likely move later. If val is still improving at epoch 40, train for more epochs rather than reading the result off a model that hasn't finished training.
* Not the same as word dropout: standard dropout zeros random features of each embedding. Word dropout (the `<UNK>` idea from #4) replaces whole words, which is what trains the `<UNK>` vector. Keep them as separate experiments.

Once dropout is set, a good next step is clipping plus an LR schedule as one stability step. After that come the bigger gains: GloVe and the unknown-word fixes.

---

**User**

> <your next message>

**AI**

<the AI's next reply>

---
