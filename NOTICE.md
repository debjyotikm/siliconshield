# Ownership and dependency notices

This repository contains SiliconShield's orchestration, training, and
integration code. Making the source visible does not grant a new license
for that code.

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
- NVIDIA PyTorch container, `nvcr.io/nvidia/pytorch:25.11-py3`: the base image
  used by `deploy/spark/glimmer/Dockerfile`.

Model weights, downloaded datasets, checkpoints, and transcripts are excluded
from this repository.
