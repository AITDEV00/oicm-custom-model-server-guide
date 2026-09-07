SHELL :=/bin/bash

.DEFAULT_GOAL=help

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-vllm: # Build custom model server image for vLLM
	@docker build --platform=linux/amd64 -t oicm/vllm:0.19.0 -f vLLM/Dockerfile vLLM

build-vllm-podman-0.22.0: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.22.0 -t oicm/vllm:0.22.0 -f vLLM-podman-0.22.0-cu129-amd64/Dockerfile vLLM-podman-0.22.0-cu129-amd64
	@podman tag oicm/vllm:0.22.0 localhost/oicm/vllm:0.22.0-cu129

build-vllm-podman-0.23.0: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.23.0 -t oicm/vllm:0.23.0 -f vLLM-podman-0.23.0-cu129-amd64/Dockerfile vLLM-podman-0.23.0-cu129-amd64
	@podman tag oicm/vllm:0.23.0 localhost/oicm/vllm:0.23.0-cu129

build-vllm-podman-0.24.0: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.24.0 -t oicm/vllm:0.24.0 -f vLLM-podman-0.24.0-cu129-amd64/Dockerfile vLLM-podman-0.24.0-cu129-amd64
	@podman tag oicm/vllm:0.24.0 localhost/oicm/vllm:0.24.0-cu129

build-vllm-podman-0.25.0: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.25.0 -t oicm/vllm:0.25.0 -f vLLM-podman-0.25.0-cu129-amd64/Dockerfile vLLM-podman-0.25.0-cu129-amd64
	@podman tag oicm/vllm:0.25.0 localhost/oicm/vllm:0.25.0-cu129

build-vllm-podman-0.25.1: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.25.1 -t oicm/vllm:0.25.1 -f vLLM-podman-0.25.1-cu129-amd64/Dockerfile vLLM-podman-0.25.1-cu129-amd64
	@podman tag oicm/vllm:0.25.1 localhost/oicm/vllm:0.25.1-cu129

build-vllm-podman-0.27.1: ## Build custom model server image for vLLM using Podman (vllm[audio] + librosa)
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.27.1 -t oicm/vllm:0.27.1 -f vLLM-podman-0.27.1-cu129-amd64/Dockerfile vLLM-podman-0.27.1-cu129-amd64
	@podman tag oicm/vllm:0.27.1 localhost/oicm/vllm:0.27.1-cu129

build-vllm-podman-minimaxm3: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 -t oicm/vllm:minimaxm3-cu129-amd64 -f vLLM-podman-minimaxm3-cu129-amd64/Dockerfile vLLM-podman-minimaxm3-cu129-amd64
	@podman tag oicm/vllm:minimaxm3-cu129-amd64 localhost/oicm/vllm:minimaxm3-cu129-amd64

build-vllm-podman-nightly: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 -t oicm/vllm:nightly-cu129-amd64 -f vLLM-podman-nightly-cu129-amd64/Dockerfile vLLM-podman-nightly-cu129-amd64
	@podman tag oicm/vllm:nightly-cu129-amd64 localhost/oicm/vllm:nightly-cu129-amd64


# save-vllm-podman: # Save vLLM Podman image to tar file for transfer
# 	@podman save -o vllm-0.22.0-cu129.tar localhost/oicm/vllm:0.22.0-cu129

# transfer-vllm-podman-0.22.0: # Transfer vLLM Podman image tar to remote server
# 	@rsync -vP --append-verify -e "ssh -c aes128-gcm@openssh.com" vllm-0.22.0-cu129.tar adeo@10.34.104.99:/home/adeo/vllm

build-sglang: # Build custom model server image for SGLang
	@docker build --platform=linux/amd64 -t oicm/sglang:0.5.9-cu129 -f sglang/Dockerfile SGLang

build-sglang-podman-0.5.14: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.14-cu129-amd64 -f SGLang-podman-v0.5.14-cu129-amd64/Dockerfile SGLang-podman-v0.5.14-cu129-amd64
	@podman tag oicm/sglang:0.5.14-cu129-amd64 localhost/oicm/sglang:0.5.14-cu129-amd64

build-sglang-podman-0.5.15: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.15.post1-cu129-amd64 -f SGLang-podman-v0.5.15-cu129-amd64/Dockerfile SGLang-podman-v0.5.15-cu129-amd64
	@podman tag oicm/sglang:0.5.15.post1-cu129-amd64 localhost/oicm/sglang:0.5.15.post1-cu129-amd64

build-sglang-podman-0.5.16: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.16-cu129-amd64 -f SGLang-podman-v0.5.16-cu129-amd64/Dockerfile SGLang-podman-v0.5.16-cu129-amd64
	@podman tag oicm/sglang:0.5.16-cu129-amd64 localhost/oicm/sglang:0.5.16-cu129-amd64

build-sglang-podman-0.5.17: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.17-cu129-amd64 -f SGLang-podman-v0.5.17-cu129-amd64/Dockerfile SGLang-podman-v0.5.17-cu129-amd64
	@podman tag oicm/sglang:0.5.17-cu129-amd64 localhost/oicm/sglang:0.5.17-cu129-amd64

build-sglang-podman-0.5.18: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.18-cu129-amd64 -f SGLang-podman-v0.5.18-cu129-amd64/Dockerfile SGLang-podman-v0.5.18-cu129-amd64
	@podman tag oicm/sglang:0.5.18-cu129-amd64 localhost/oicm/sglang:0.5.18-cu129-amd64

build-sglang-podman-0.5.19: # Build custom model server image for SGLang using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.19-cu129-amd64 -f SGLang-podman-v0.5.19-cu129-amd64/Dockerfile SGLang-podman-v0.5.19-cu129-amd64
	@podman tag oicm/sglang:0.5.19-cu129-amd64 localhost/oicm/sglang:0.5.19-cu129-amd64

build-sglang-podman-glm-5.3-flash: # Build custom model server image for SGLang (GLM-5.3-Flash) using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:glm-5.3-flash-cu129-amd64 -f SGLang-podman-glm-5.3-flash-cu129-amd64/Dockerfile SGLang-podman-glm-5.3-flash-cu129-amd64
	@podman tag oicm/sglang:glm-5.3-flash-cu129-amd64 localhost/oicm/sglang:glm-5.3-flash-cu129-amd64

build-sglang-podman-qwen38flashnext: # Build custom model server image for SGLang (Qwen3.8-FlashNext) using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:qwen38flashnext-cu129-amd64 -f SGLang-podman-qwen38flashnext-cu129-amd64/Dockerfile SGLang-podman-qwen38flashnext-cu129-amd64
	@podman tag oicm/sglang:qwen38flashnext-cu129-amd64 localhost/oicm/sglang:qwen38flashnext-cu129-amd64

build-cputests: # Build CPU + Silero-VAD diagnostic image
	@podman build --platform=linux/amd64 -t cpu-vad-diag:latest -f cputests/Dockerfile cputests

build-diagnostics: # Build lightweight OICM diagnostic image
	@podman build --platform=linux/amd64 -t oicm/diagnostics:latest -f diagnostics/Dockerfile diagnostics


# podman save -o diagnostics-oicm.tar localhost/oicm/diagnostics:latest
# rsync -vP --append-verify -e "ssh -c aes128-gcm@openssh.com" diagnostics-oicm.tar adeo@10.34.104.99:/home/adeo
build-vllm-podman-vibevoice-awq: ## Build custom model server image for VibeVoice ASR (AWQ int4) on vLLM 0.27.1
	@podman build --platform=linux/amd64 -t oicm/vllm:vibevoice-awq-0.27.1 -f vLLM-podman-vibevoice-awq-0.27.1-cu129-amd64/Dockerfile vLLM-podman-vibevoice-awq-0.27.1-cu129-amd64
	@podman tag oicm/vllm:vibevoice-awq-0.27.1 localhost/oicm/vllm:vibevoice-awq-0.27.1-cu129

# Run VibeVoice ASR locally (GPU, host network). EXTRA_ARGS follow the model
# card recommendations: --trust-remote-code and --dtype float16 are INJECTED BY
# startup.sh (see image), so they are not repeated here. We deliberately do NOT
# pass --quantization awq_marlin: vLLM auto-promotes AWQ to Marlin on SM90/SM120
# (verified in engine log: "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod").
# --no-enable-prefix-caching matches upstream: the plugin salts every audio item
# with a random value to bypass prefix caching, so enabling it is pure overhead.
run-vllm-podman-vibevoice-awq: ## Run VibeVoice ASR container locally (GPU)
	@podman rm -f vibevoice-asr >/dev/null 2>&1 || true
	@podman run -d --name vibevoice-asr \
	  --device nvidia.com/gpu=all --security-opt label=disable \
	  --network host \
	  -v /srv/shared/download/hf/lemuriandezapada/VibeVoice-ASR-awq-int4:/pvc-home/model:ro,Z \
	  -e INIT_CONTAINER_USE_PVC=True -e PVC_PATH=/pvc-home \
	  -e MODEL_ID=vibevoice \
	  -e EXTRA_ARGS="--max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --enable-chunked-prefill --port 8321" \
	  localhost/oicm/vllm:vibevoice-awq-0.27.1-cu129
	@echo "vibevoice-asr starting on http://localhost:8321 (model load ~3.5 min)"
	@echo "follow with: podman logs -f vibevoice-asr"

benchmark-vibevoice-awq: ## Run RTF benchmark against a running vibevoice-asr container
	@python3 vLLM-podman-vibevoice-awq-0.27.1-cu129-amd64/rtf_benchmark.py \
	  --base-url http://localhost:8321 \
	  --audio-dir /home/jyao/ADEO/vibevoice.cpp/audio --runs 3 --warmup 1
