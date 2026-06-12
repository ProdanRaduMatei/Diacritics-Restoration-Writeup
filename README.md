# Romanian Diacritics Restoration

Local, on-premises pipeline for restoring Romanian diacritics.

The final inference path is:

```text
input fara diacritice
  -> normalization
  -> SentencePiece + lightweight encoder-decoder Transformer
  -> base-text safety constraint
  -> lexicon safety constraint
  -> dictionary fallback when unsafe
  -> final output
```

## Setup

The project uses the local `.venv` already created in this workspace:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download ro_core_news_sm
```

The current workspace already has:

- `sentencepiece==0.2.1`
- `spacy==3.8.14`
- `ro_core_news_sm==3.8.0`
- `torch==2.7.0`

## Reproduce Artifacts

```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train_baselines.py
.venv/bin/python scripts/train_transformer.py --preset small --epochs 20 --device cpu
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/benchmark.py --repeats 50
```

The trained local artifacts are:

- `artifacts/model/transformer.pt`
- `artifacts/tokenizer/ro_diacritics.model`
- `artifacts/baselines/frequency_dictionary.json`
- `artifacts/baselines/ngram_lm.json`

## Inference

```bash
.venv/bin/python scripts/restore_text.py "Cum sa-ti poti inchipui, de pilda, un oras"
```

Expected output:

```text
Cum să-ți poți închipui, de pildă, un oraș
```

To see which layer produced the answer:

```bash
.venv/bin/python scripts/restore_text.py --source "Cum sa-ti poti inchipui, de pilda, un oras"
```

## Results

Test split: 664 document-level samples.

| system | char acc | word acc | diacritic-pos acc | sentence exact | constraint violations |
|---|---:|---:|---:|---:|---:|
| frequency dictionary | 0.9856 | 0.9423 | 0.9547 | 0.4669 | 0.0030 |
| n-gram reranker | 0.9716 | 0.8908 | 0.9111 | 0.1958 | 0.0030 |
| transformer + safety + fallback | 0.9856 | 0.9421 | 0.9548 | 0.4714 | 0.0000 |

CPU benchmark on short sentences:

| mean ms | p50 ms | p95 ms | model MB | dictionary MB | ngram MB |
|---:|---:|---:|---:|---:|---:|
| 20.299 | 21.300 | 23.388 | 15.054 | 0.453 | 7.376 |

Detailed reports:

- `reports/cleaning_report.txt`
- `reports/evaluation.json`
- `reports/hard_cases.md`
- `reports/linguistic_report.md`
- `reports/benchmark.json`

