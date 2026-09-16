<!--
SPDX-FileCopyrightText: 2023 Deutsche Telekom AG

SPDX-License-Identifier: CC0-1.0    
-->

# Anomaly Detection Framework (ADF)

[![REUSE Compliance Check](../../actions/workflows/reuse-compliance.yml/badge.svg)](../../actions/workflows/reuse-compliance.yml)

## About

ADF is a Python library for detecting anomalies in machine data, in log lines and in metric time
series, with models small enough to train and to run on a single machine.

The main use cases of ADF are
- reducing raw log lines to templates with a Drain parser that is extended online when an unseen
  message arrives, so that a new event is never silently dropped;
- adapting a compact sentence transformer to the logs of one system, so that the semantics of a
  message, and not only its identifier, decide how it is scored;
- scoring windows of log lines with an LSTM autoencoder trained on normal data only, and turning the
  reconstruction error into a verdict for every line or session;
- detecting anomalies and input drift in metric time series with the same building blocks.

## Demos

[`public_demos/`](public_demos/README.md) holds three notebooks that run the whole pipeline end to end
on public log data sets. Each one starts from the raw log, recomputes every score and checks the
result against the shipped configuration:

| Data set | Verdict per | Test set | Precision | Recall | F1 |
|---|---|---|---|---|---|
| [HDFS v1](public_demos/HDFS/hdfs.ipynb) | block | 98,888 blocks, 15,154 anomalous | 0.99279 | 0.99947 | 0.99612 |
| [BGL](public_demos/BGL/bgl.ipynb) | line | 949,593 lines, 46,365 anomalous | 0.98449 | 0.98184 | 0.98317 |
| [Thunderbird](public_demos/Thunderbird/thunderbird.ipynb), first 20 M lines | line | 4,000,000 lines, 283,846 anomalous | 0.99429 | 0.99040 | 0.99234 |

The thresholds are chosen without test data, and the demos README explains the protocol, what is
deterministic and what is not.

## Installation

ADF needs Python 3.11 and [uv](https://docs.astral.sh/uv/). In the repository root:

```bash
uv sync --locked --extra deploy
```

The `deploy` extra installs the training and inference stack (PyTorch, Lightning,
sentence-transformers). Use `--extra core` for the lighter set without the deep learning stack, or
`--extra all` for everything including the development tools.

To open the demo notebooks:

```bash
uv run --with jupyter jupyter lab
```

## Repository layout

```
src/adf/core       building blocks shared by both pipelines: data frames, Lightning wiring, evaluation
src/adf/logs       log pipeline: Drain parsing, template embeddings, autoencoders, inference
src/adf/metrics    metric pipeline: models, input drift monitoring, preprocessing
public_demos/      three reproducible demos on public log data sets, with their artefacts
tests/             the test suite
```

## Development

```bash
uv sync --locked --extra all
```

The checks that have to pass are the ones in `.pre-commit-config.yaml`: `ruff format` and
`ruff check` on `src`, `mypy --strict` on `src`, and `pytest`.

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/) in version 2.1 as our code of conduct. Please see the details in our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All contributors must abide by the code of conduct.

By participating in this project, you agree to abide by its [Code of Conduct](./CODE_OF_CONDUCT.md) at all times.

## Licensing
Copyright (c) 2026 Deutsche Telekom AG

All content in this repository is licensed under at least one of the licenses found in [./LICENSES](./LICENSES); you may not use this file, or any other file in this repository, except in compliance with the Licenses. 
You may obtain a copy of the Licenses by reviewing the files found in the [./LICENSES](./LICENSES) folder.

Unless required by applicable law or agreed to in writing, software distributed under the Licenses is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See in the [./LICENSES](./LICENSES) folder for the specific language governing permissions and limitations under the Licenses.

This project follows the [REUSE standard for software licensing](https://reuse.software/). 
Each file contains copyright and license information, and license texts can be found in the [./LICENSES](./LICENSES) folder. For more information visit https://reuse.software/.
You can find a guide for developers at https://telekom.github.io/reuse-template/.
