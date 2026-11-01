# Ownership and dependency notices

SiliconShield's orchestration, training, and integration code in this
repository is released under the Apache License, Version 2.0 (see
LICENSE). The SiliconShield RTL benchmark dataset is likewise offered
under Apache-2.0.

Dataset catalogs, source-derived evaluation cases, and experiment recipes are
loaded from local files. The repository includes only illustrative examples.

Dependencies retain their own licenses and terms:

- LAYA: [original project](https://github.com/NandhaKishorM/laya) and
  [LAYA Next](https://github.com/debjyotikm/laya-next). LAYA core is an external
  dependency; this repository contains the security-training scripts.
- [Google Research Harness²](https://github.com/google-research/harness2):
  pinned in `requirements/harness2.txt` and installed separately.
- [Meta Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B):
  weights and tokenizer must be downloaded separately.
- OpenTitan and HardSecBench: upstream RTL used to construct scoped contracts
  and verified variants. Source-derived material retains its upstream terms.
- Icarus Verilog, Verilator, and Yosys: compilation, simulation, and solver
  checks. These tool results are separate from decision-model predictions.
- NVIDIA PyTorch container, `nvcr.io/nvidia/pytorch:25.11-py3`: the base image
  used by `deploy/spark/glimmer/Dockerfile`.

Model weights, downloaded datasets, checkpoints, and transcripts are excluded
from this repository.

The owner has selected Apache-2.0 for the SiliconShield Contract-1 weight
release, consistent with Contract-2. Foundation weights and source-derived
material retain their upstream terms. The owner decision is distinct from
publication of corrected Hub metadata. See docs/releases.md for the pinned
artifacts and synchronization status.
