ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir transformers==5.21.0 peft==0.21.2 \
    tokenizers==0.23.3 safetensors==0.8.0
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
WORKDIR /workspace
ENTRYPOINT []
