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

build-sglang-podman-0.5.19-cu130: # Build custom model server image for SGLang using Podman (CUDA 13.0 base, R580-era compat libs)
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.19-cu130-amd64 -f SGLang-podman-v0.5.19-cu130-amd64/Dockerfile SGLang-podman-v0.5.19-cu130-amd64
	@podman tag oicm/sglang:0.5.19-cu130-amd64 localhost/oicm/sglang:0.5.19-cu130-amd64

build-sglang-podman-v0.5.19-glm53-dcp: # Build GLM-5.x DSA/DCP image (v0.5.19-cu130 + only PR #31821, frozen patch)
	@podman build --platform=linux/amd64 -t oicm/sglang:v0.5.19-cu130-glm53-dcp-f1eb59f54f-amd64 -f SGLang-podman-v0.5.19-cu130-amd64-glm53-dcp/Dockerfile SGLang-podman-v0.5.19-cu130-amd64-glm53-dcp
	@podman tag oicm/sglang:v0.5.19-cu130-glm53-dcp-f1eb59f54f-amd64 localhost/oicm/sglang:v0.5.19-cu130-glm53-dcp-f1eb59f54f-amd64

build-sglang-podman-v0.5.19-kimi-k3: # Build Kimi-K3 kv-free fix image (v0.5.19-cu130 + only PR #38982, frozen patch)
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.19-cu130-amd64-kimi-k3 -f SGLang-podman-v0.5.19-cu130-amd64-kimi-k3/Dockerfile SGLang-podman-v0.5.19-cu130-amd64-kimi-k3
	@podman tag oicm/sglang:0.5.19-cu130-amd64-kimi-k3 localhost/oicm/sglang:0.5.19-cu130-amd64-kimi-k3

build-sglang-podman-glm-5.3-flash: # Build custom model server image for SGLang (GLM-5.3-Flash) using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:glm-5.3-flash-cu129-amd64 -f SGLang-podman-glm-5.3-flash-cu129-amd64/Dockerfile SGLang-podman-glm-5.3-flash-cu129-amd64
	@podman tag oicm/sglang:glm-5.3-flash-cu129-amd64 localhost/oicm/sglang:glm-5.3-flash-cu129-amd64

build-sglang-podman-qwen38flashnext: # Build custom model server image for SGLang (Qwen3.8-FlashNext) using Podman
	@podman build --platform=linux/amd64 -t oicm/sglang:qwen38flashnext-cu129-amd64 -f SGLang-podman-qwen38flashnext-cu129-amd64/Dockerfile SGLang-podman-qwen38flashnext-cu129-amd64
	@podman tag oicm/sglang:qwen38flashnext-cu129-amd64 localhost/oicm/sglang:qwen38flashnext-cu129-amd64

build-sglang-podman-dev-dsv41: # Build SGLang dev-dsv41 image (DeepSeek V4.1, cu130, lmsysorg/sglang:dev-dsv41 base)
	@podman build --platform=linux/amd64 -t oicm/sglang:dev-dsv41-cu130-amd64 -f SGLang-podman-dev-dsv41-cu130-amd64/Dockerfile SGLang-podman-dev-dsv41-cu130-amd64
	@podman tag oicm/sglang:dev-dsv41-cu130-amd64 localhost/oicm/sglang:dev-dsv41-cu130-amd64

build-sglang-podman-0.5.21-cu130: # Build custom model server image for SGLang v0.5.21 (CUDA 13.0 base, quiet startup harness)
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.21-cu130-amd64 -f SGLang-podman-v0.5.21-cu130-amd64/Dockerfile SGLang-podman-v0.5.21-cu130-amd64
	@podman tag oicm/sglang:0.5.21-cu130-amd64 localhost/oicm/sglang:0.5.21-cu130-amd64

push-sglang-podman-0.5.21-cu130: # Tag + push SGLang v0.5.21 cu130 image to Al Ain Harbor
	@podman tag localhost/oicm/sglang:0.5.21-cu130-amd64 registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.21-cu130-amd64
	@podman push --tls-verify=false registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.21-cu130-amd64

# Pre-staged for the v0.5.22 release (harness is version-agnostic; only the
# upstream base tag in the Dockerfile's FROM line moves). If lmsysorg has not
# published v0.5.22-cu130 yet, the build will fail at the FROM pull -- adjust
# that single line, nothing else, once the tag exists.
build-sglang-podman-0.5.22-cu130: # Build custom model server image for SGLang v0.5.22 (CUDA 13.0 base, quiet startup harness)
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.22-cu130-amd64 -f SGLang-podman-v0.5.22-cu130-amd64/Dockerfile SGLang-podman-v0.5.22-cu130-amd64
	@podman tag oicm/sglang:0.5.22-cu130-amd64 localhost/oicm/sglang:0.5.22-cu130-amd64

push-sglang-podman-0.5.22-cu130: # Tag + push SGLang v0.5.22 cu130 image to Al Ain Harbor
	@podman tag localhost/oicm/sglang:0.5.22-cu130-amd64 registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.22-cu130-amd64
	@podman push --tls-verify=false registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.22-cu130-amd64

build-sglang-podman-0.5.23-cu130: # Build custom model server image for SGLang v0.5.23 (CUDA 13.0 base, quiet startup harness)
	@podman build --platform=linux/amd64 -t oicm/sglang:0.5.23-cu130-amd64 -f SGLang-podman-v0.5.23-cu130-amd64/Dockerfile SGLang-podman-v0.5.23-cu130-amd64
	@podman tag oicm/sglang:0.5.23-cu130-amd64 localhost/oicm/sglang:0.5.23-cu130-amd64

# The completed 2026-11-11 artifact did NOT use the old OCI-directory relay.
# This target is an intentional first-HF-publication entry point: an existing
# worktree tar is checked by expected raw SHA256 and source image config before
# admission. It never rebuilds an unrelated image over the completed evidence.
deliver-sglang-podman-0.5.23-cu130: # Export existing image and publish its Docker archive to the authorized public HF relay
	@SGLang-podman-v0.5.23-cu130-amd64/publish_docker_archive.py

continue-sglang-podman-0.5.23-cu130: # Read HF→MinIO evidence, optionally resume node→Harbor; never launch another live writer
	@SGLang-podman-v0.5.23-cu130-amd64/complete_harbor.py $(ARGS)

verify-sglang-podman-0.5.23-cu130: # Read-only Harbor manifest/config readback (no upload/import/push/deployment)
	@SGLang-podman-v0.5.23-cu130-amd64/verify-alain.py --insecure-registry $(ARGS)

push-sglang-podman-0.5.23-cu130: # Direct workstation push (historical/small/cached only; protected auth preflight required)
	@REGISTRY='registry.adeoaiengine.ecouncil.ae' DIRECT_PUSH_CONFIRMED_ALLOW_SLOW=required EXTRA_FLAGS='--tls-verify=false' \
	SGLang-podman-v0.5.23-cu130-amd64/harbor-auth.sh push-only localhost/oicm/sglang:0.5.23-cu130-amd64

# Kimi-K3.1-Pro. NOT a v0.5.x release build: no tagged SGLang release can load
# this model (v0.5.23 has no KimiK31ForConditionalGeneration and no 'kimi_k31'
# model_type). Derived from the official preview image, which upstream builds
# from the kimi-k3.1-day-0 branch with docker/k31.Dockerfile (10 FlashMLA ops,
# FlashInfer DynBlock at 1024 experts, Rust TreeCore, Prims-TS, DeepGEMM,
# patched Dynamo). See the example's README.md.
build-sglang-podman-kimi-k31-pro: # Build Kimi-K3.1-Pro (DFLASH) serving image from lmsysorg/sglang:dev-kimi-k31-pro
	@podman build --platform=linux/amd64 -t oicm/sglang:kimi-k31-pro-cu130-amd64 -f SGLang-podman-kimi-k31-pro-cu130-amd64/Dockerfile SGLang-podman-kimi-k31-pro-cu130-amd64
	@podman tag oicm/sglang:kimi-k31-pro-cu130-amd64 localhost/oicm/sglang:kimi-k31-pro-cu130-amd64

verify-sglang-podman-kimi-k31-pro: # Re-run all 3 K3.1 build gates against the built image (no GPU needed)
	@echo '--- gate 1: compiled K3.1 components (upstream verify_image.py) ---'
	@podman run --rm -e HOME=/tmp --entrypoint python3 oicm/sglang:kimi-k31-pro-cu130-amd64 /opt/k31-image/verify_image.py
	@echo '--- gate 2: KimiK31 model class + config registration ---'
	@podman run --rm -e HOME=/tmp --entrypoint python3 oicm/sglang:kimi-k31-pro-cu130-amd64 -c "from sglang.srt.models.kimi_k3 import KimiK31ForConditionalGeneration as K; print('  ', K.__name__)"
	@echo '--- gate 3: DFLASH flag set in the argparser ---'
	@podman run --rm -e HOME=/tmp --entrypoint python3 oicm/sglang:kimi-k31-pro-cu130-amd64 /app/k31_flags_check.py

# Delivery goes through MinIO, NOT podman push (same reasoning as omni-dev: the
# direct path is 0.14 MB/s per TCP connection and re-uploads every layer
# uncompressed). This image's Dockerfile puts the OICM harness LAST as a single
# COPY, so a script edit rebuilds in ~6 s / 2 layers; deliver-image.sh then
# moves only the changed blob. First delivery ~35 min, re-delivery ~3 min.
deliver-sglang-podman-kimi-k31-pro: build-sglang-podman-kimi-k31-pro # Ship K3.1 image to Al Ain Harbor via MinIO relay
	@SGLang-podman-kimi-k31-pro-cu130-amd64/deliver-image.sh \
		localhost/oicm/sglang:kimi-k31-pro-cu130-amd64 \
		openinnovationai/platform/mlops/mlops-serving/sglang \
		kimi-k31-pro-cu130-amd64

# Only correct when every layer already exists in Harbor. Prefer deliver-* .
push-sglang-podman-kimi-k31-pro: # Tag + podman push K3.1 image (slow; prefer deliver)
	@podman tag localhost/oicm/sglang:kimi-k31-pro-cu130-amd64 registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:kimi-k31-pro-cu130-amd64
	@podman push --tls-verify=false registry.adeoaiengine.issue.acme/openinnovationai/platform/mlops/mlops-serving/sglang:kimi-k31-pro-cu130-amd64

build-sglang-podman-omni-dev: # Build SGLang-Omni v0.2.0 serving image (cu130, lmsysorg/sglang 0.5.22 base)
	@podman build --platform=linux/amd64 -t oicm/sglang-omni:0.2.0-cu130-amd64 -f SGLang-podman-omni-dev-cu130-amd64/Dockerfile SGLang-podman-omni-dev-cu130-amd64
	@podman tag oicm/sglang-omni:0.2.0-cu130-amd64 localhost/oicm/sglang-omni:0.2.0-cu130-amd64

# Delivery goes through MinIO, NOT podman push. Direct push measured 2.2 MB/s
# from this box and re-uploads every layer uncompressed (0 percent dedupe even
# against layers Harbor already has as gzip): ~5 h for this image. The relay is
# ~60 min because the node talks to Harbor in-cluster.
deliver-sglang-podman-omni-dev: build-sglang-podman-omni-dev # Ship SGLang-Omni image to Al Ain Harbor via S3 relay (~60 min)
	@SGLang-podman-omni-dev-cu130-amd64/deliver-image.sh \
	        localhost/oicm/sglang-omni:0.2.0-cu130-amd64 \
	        openinnovationai/platform/mlops/mlops-serving/sglang-omni \
                0.2.0-cu130-amd64

# Only correct when the layers already exist in Harbor. Prefer deliver-* instead.
push-sglang-podman-omni-dev: # Tag + podman push SGLang-Omni image (slow; prefer deliver)
	@podman tag localhost/oicm/sglang-omni:0.2.0-cu130-amd64 registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang-omni:0.2.0-cu130-amd64
	@podman push --tls-verify=false registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang-omni:0.2.0-cu130-amd64

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
