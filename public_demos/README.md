<!--
SPDX-FileCopyrightText: 2026 Deutsche Telekom AG

SPDX-License-Identifier: CC-BY-4.0
-->

# Anomaly Detection Framework (ADF) — demos

Three notebooks that run the Anomaly Detection Framework (ADF) end to end on public log datasets.
Each one starts from the raw log, recomputes every score and checks the outcome against the
published result:

| Dataset | Verdict per | Test set | Precision | Recall | F1 |
|---|---|---|---|---|---|
| [HDFS v1](HDFS/hdfs.ipynb) | block | 98,888 blocks, 15,154 anomalous | 0.99279 | 0.99947 | 0.99612 |
| [BGL](BGL/bgl.ipynb) | line | 949,593 lines, 46,365 anomalous | 0.98449 | 0.98184 | 0.98317 |
| [Thunderbird](Thunderbird/thunderbird.ipynb), first 20 M lines | line | 4,000,000 lines, 283,846 anomalous | 0.99429 | 0.99040 | 0.99234 |

## How it works

All three notebooks follow the same pipeline:

1. **Drain** turns log messages into templates. Masking rules replace values such as numbers,
   paths and addresses with placeholders before clustering.
2. A **sentence transformer**, `all-MiniLM-L6-v2` fine-tuned on the templates with SimCSE, turns
   each template into a 384-dimensional vector.
3. An **LSTM autoencoder**, trained only on normal windows of consecutive log lines, learns to
   reconstruct them. Its reconstruction error is the anomaly score, and a score above the threshold
   makes the verdict anomalous.

HDFS is split at random by block, and every block gets one verdict. BGL and Thunderbird are split
chronologically, windows are built within each node, and every line gets its own verdict. The
threshold is never chosen on test data (see [Thresholds](#thresholds)). Each notebook explains its
protocol in detail.

## What is in each folder

```
public_demos/
├── HDFS/          hdfs.ipynb         config.yaml   artifacts/
├── BGL/           bgl.ipynb          config.yaml   artifacts/
└── Thunderbird/   thunderbird.ipynb  config.yaml   artifacts/
```

- **The notebook** contains the whole pipeline, saved with the outputs of a complete run.
- **`config.yaml`** holds every parameter, the dataset checksums, the threshold and the expected
  result. The notebook reads its settings from this file.
- **`artifacts/`** holds everything that takes long to compute: the fitted Drain parser
  (`drain/state.bin`, `drain/config.ini`), the template vocabulary (`templates.json`), the template
  embeddings (`template_embeddings.npy`) and the trained autoencoder (`autoencoder.ckpt`). BGL and
  Thunderbird also ship `threshold_templates.json` and `threshold_embeddings.npy`: the templates and
  vectors of the training and validation lines the parser does not know, which a model trained with
  the shipped embeddings needs to choose its threshold. Each notebook checks the checksums of all of
  them against `config.yaml` before it starts.

The datasets are not part of the repository.

## Setup

**1. The Python environment.** Python 3.11 and [uv](https://docs.astral.sh/uv/). In the repository
root:

```bash
uv sync --locked --extra deploy
uv run --with jupyter jupyter lab
```

**2. The datasets.** Each notebook reads its data from `WORK_DIR/data/`. `WORK_DIR` is set in the
first code cell (default `~/adf_demos/<dataset>`), and everything the notebook computes is written
there too. The files are checked against the checksums in `config.yaml`.

| Dataset | Source | Prepare |
|---|---|---|
| HDFS v1 | [`HDFS_v1.zip`](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1), 187 MB, from the [LogHub](https://github.com/logpai/loghub) collection on Zenodo | `unzip HDFS_v1.zip -d WORK_DIR/data/HDFS_v1` |
| BGL | [`bgl2.gz`](http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz), 63 MB, from the [USENIX Computer Failure Data Repository](https://www.usenix.org/cfdr-data) | `gunzip -c bgl2.gz > WORK_DIR/data/bgl2` |
| Thunderbird | [`tbird2.gz`](http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz), 2 GB, from the [USENIX Computer Failure Data Repository](https://www.usenix.org/cfdr-data) | `gunzip -c tbird2.gz \| head -n 20000000 > WORK_DIR/data/tbird2_20M` |

**3. The base sentence transformer**, needed only when you fine-tune the embeddings yourself:
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The notebook downloads it into
`WORK_DIR/models/`.

## Running a notebook

Three switches in the first code cell decide what is loaded and what is recomputed:

| Switch | `"ours"` (default) | The alternative |
|---|---|---|
| `DRAIN` | load the shipped parser | `"fit"` — fit Drain here |
| `EMBEDDINGS` | load the shipped template embeddings | `"finetune"` — fine-tune the sentence transformer here |
| `AUTOENCODER` | load the shipped model | `"train"` — train with the parameters in `config.yaml` |

With all three set to `"ours"`, the notebook reproduces the published numbers exactly. The
evaluation section, 6.5 in HDFS and 7.4 in BGL and Thunderbird, prints `REPRODUCED` when the true
positives, false positives and false negatives match `config.yaml`.

Reproducing a result takes about 45 minutes for HDFS, most of it spent scoring the test blocks,
and one to two minutes each for BGL and Thunderbird. Training your own autoencoder takes about two
hours for BGL, seven for HDFS and eleven for Thunderbird; for HDFS, choosing the threshold of your
own model adds about half an hour of scoring validation blocks. All times are from an Apple M4 Max
laptop.

## Reproducibility

> [!WARNING]
> **Training the autoencoder on Apple Silicon (MPS) is not deterministic.** Two training runs with
> the same data, parameters and seed produce different models. A model you train yourself gives
> scores close to the published ones but not identical, and the notebook chooses its threshold
> again, the same way as for the shipped model.
> The precision, recall and F1 you get will be close to the published numbers but not the same.
> This is why the trained models are shipped. To reproduce the published numbers, keep
> `AUTOENCODER = "ours"`.

Everything else was checked on the environment pinned in `uv.lock` (macOS on Apple Silicon,
Python 3.11, torch 2.5.1, lightning 2.5.5, sentence-transformers 3.2.0, transformers 4.45.2):

- **Scoring with the shipped model** gives the same scores every time on the same device. BGL and
  Thunderbird score on the first accelerator available (MPS, then CUDA, then the CPU), and HDFS
  scores on the CPU. Between MPS and the CPU the scores differ by at most 1.3·10⁻⁸, and not a
  single verdict changes. CUDA has not been tested.
- **Fitting Drain** is deterministic. For BGL and Thunderbird it rebuilds the shipped parser,
  template ids included. For HDFS it finds the same templates but can number them differently. The
  shipped embeddings and model follow the shipped numbering, so a parser fitted from scratch for
  HDFS needs its own embeddings and autoencoder too.
- **Fine-tuning the sentence transformer** is deterministic on the same hardware and environment:
  the replay produced the shipped embeddings bit for bit. It has not been tested on other CPU
  architectures.

### Thresholds

No threshold is chosen on test data. `threshold_selector` takes scored data with labels and returns
the threshold that separates the two classes best (the highest F1), placed midway between two
neighbouring scores. It runs on:

- **BGL and Thunderbird:** every line of the training and validation periods, scored by the same
  rule as a test line. Training keeps only normal windows, but the labelled lines of these periods
  are still in the log. Lines the parser does not know are first fitted into a copy of it that has
  seen no test line.
- **HDFS:** the normal validation blocks and a seeded 10 % of the anomalous blocks. No training or
  validation block is anomalous, so these anomalous blocks are set aside for the threshold and left
  out of the evaluation.

The thresholds in `config.yaml` were chosen this way for the shipped models, and a model trained in
a notebook gets its own, chosen the same way.

Identical windows get identical scores, so scores come in groups of equal values. Keep the
thresholds at full precision: rounding one can move a whole group across it.
