# Diacritics Restoration Writeup

## Problem Understanding

The task is Romanian diacritics restoration: map text without diacritics to the same text with diacritics. I treated it as sequence-to-sequence, not pure character substitution, because words such as `peste`, `fata`, `mana`, `para`, `casa` are context-dependent. The system still needs a hard constraint: the output must not rewrite the user's text.

Alternatives considered:

- Character-level replacement/classification: very safe, but weak on contextual ambiguity.
- Frequency dictionary: strong and tiny, but picks the most common form even when context disagrees.
- N-gram reranker: can use context, but on this small noisy corpus it overfits local statistics and performed worse than the dictionary.
- Encoder-decoder Transformer: flexible contextual model, but can rewrite text unless constrained.

Final choice: a hybrid. I trained a lightweight seq2seq Transformer, then wrapped it in strict safety layers. If the model rewrites text or proposes an unseen weird form such as `Câsa`, the system falls back to the conservative dictionary.

## Data and Cleaning

The corpus has 6646 `.gt.txt` / `.gt.txt.bak` pairs. Cleaning was conservative:

- normalized Romanian comma-below characters: `ş/ţ` -> `ș/ț`
- removed exact duplicate pairs
- rejected empty/corrupted/replacement-char rows
- rejected major length or base-text mismatches
- split by document, not random lines

Result:

- kept: 6635
- removed duplicates: 11
- train/valid/test: 5308 / 663 / 664

I deliberately did not correct OCR-looking text such as `fn` for `în`, because those errors appear in both source and target and changing them would turn the task into OCR correction, not diacritics restoration.

## Model and Deployment Decision

The trained artifact uses the `small` preset:

- SentencePiece BPE vocab: 4000
- encoder layers: 3
- decoder layers: 3
- hidden size: 192
- heads: 4
- feed-forward size: 768
- parameters: 3,936,928
- checkpoint size: 15.054 MB

I implemented a larger `recommended` preset too (`4/4/256/1024`, vocab 8000), but chose `small` for the trained artifact because the target deployment is modest on-prem hardware and the baseline is already strong. The larger model is a reasonable next run if GPU or longer CPU time is available.

Training command used:

```bash
.venv/bin/python scripts/train_transformer.py --preset small --epochs 20 --device cpu
```

Training took 254.945 seconds on CPU. Best validation loss at epoch 20 was 1.3089.

## Safety and Fallback

Two constraints are applied after decoding:

1. Base-text constraint:
   `remove_diacritics(output) == normalized_input`

2. Lexicon constraint:
   for each generated word, if the base form was seen in training, the diacritized output must be one of the observed forms for that base.

The second rule is important. The Transformer can produce base-preserving but wrong forms such as `Răymond`, `Câsa`, or `pără`. These pass the first check but fail the lexicon check.

Fallback order is conservative by default:

```text
Transformer -> safety checks -> dictionary fallback -> identity
```

The n-gram reranker remains implemented and benchmarked, but is not the default fallback because it underperformed.

## Results

| system | char acc | word acc | diacritic-pos acc | sentence exact | constraint violations |
|---|---:|---:|---:|---:|---:|
| frequency dictionary | 0.9856 | 0.9423 | 0.9547 | 0.4669 | 0.0030 |
| n-gram reranker | 0.9716 | 0.8908 | 0.9111 | 0.1958 | 0.0030 |
| transformer + safety + fallback | 0.9856 | 0.9421 | 0.9548 | 0.4714 | 0.0000 |

The final pipeline accepted the Transformer output on 182/664 test samples, used dictionary fallback on 480, and identity fallback on 2. The hybrid only slightly improves automatic scores over the dictionary, but it removes constraint violations and creates a safe place to improve the contextual model later.

CPU benchmark on short inputs:

- mean latency: 20.299 ms
- p50: 21.300 ms
- p95: 23.388 ms
- model size: 15.054 MB

## Hard Cases

The manual hard cases are in `reports/hard_cases.md`.

Examples:

- `Am pus peste peste legume.` -> noun vs preposition. The model still leaves `peste peste`; this needs stronger semantic context.
- `Fata sta in fata casei.` -> `fata/față/fața`; current output is `Fața sta în fața casei`, showing dictionary bias.
- `Mana dreapta era in mana medicului.` -> indefinite vs definite noun; current output misses `dreaptă`.
- `Sa-si ia cartea sau sa o lase?` -> clitic punctuation; model gets `-și` and `să`, but misses initial `Să`.

These are useful because they expose the exact boundary between lexical frequency and real contextual understanding.

## Linguistic Layer

spaCy with `ro_core_news_sm` is installed and used for auxiliary reporting in `reports/linguistic_report.md`. I use it for ambiguity inspection and error analysis, not for core inference. RoWordNet/NLP-Cube are treated as optional hooks; I would add them for better candidate generation and semantic grouping, but not as hard runtime dependencies.

## AI Usage Journal

I used AI assistance for:

- brainstorming the hybrid architecture and deciding where safety constraints belong
- drafting the first code skeleton
- generating candidate hard cases
- stress-testing the n-gram idea conceptually

Where AI was misleading or too simple:

- A naive suggestion was "just use character-level restoration"; I rejected it because it cannot resolve `peste/pește` or `fata/fața` reliably.
- The first n-gram reranker looked more sophisticated than the dictionary but scored worse. I kept it as an experiment and changed the production fallback to dictionary-first.
- The first Transformer checkpoints seemed to "work" by loss, but raw outputs showed text rewrites. The safety and lexicon constraints were added after inspecting those failures manually.

Manual decisions I made:

- conservative cleaning instead of OCR repair
- document-level split
- small trained model instead of the bigger recommended preset
- dictionary-first fallback based on test evidence
- lexicon safety constraint to reject base-preserving hallucinations

## Limitations and Next Steps

The Transformer is under-trained relative to what the architecture can do. It improves sentence exact match slightly, but hard cases remain weak. With more time I would:

- train the recommended preset and compare against `small`
- add a confidence score instead of binary lexicon acceptance
- add constrained decoding or token-level edit projection
- train a contextual token classifier as a safety-preserving companion model
- use RoWordNet/NLP-Cube for ambiguity sets and semantic hard-case mining
- quantize/export to ONNX after the model is demonstrably stronger

