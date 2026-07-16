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

build-cputests: # Build CPU + Silero-VAD diagnostic image
	@podman build --platform=linux/amd64 -t cpu-vad-diag:latest -f cputests/Dockerfile cputests

build-diagnostics: # Build lightweight OICM diagnostic image
	@podman build --platform=linux/amd64 -t oicm/diagnostics:latest -f diagnostics/Dockerfile diagnostics


# podman save -o diagnostics-oicm.tar localhost/oicm/diagnostics:latest
# rsync -vP --append-verify -e "ssh -c aes128-gcm@openssh.com" diagnostics-oicm.tar adeo@10.34.104.99:/home/adeo