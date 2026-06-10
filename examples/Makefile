SHELL :=/bin/bash

.DEFAULT_GOAL=help

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-vllm: # Build custom model server image for vLLM
	@docker build --platform=linux/amd64 -t oicm/vllm:0.19.0 -f vLLM/Dockerfile vLLM

build-vllm-podman: # Build custom model server image for vLLM using Podman
	@podman build --platform=linux/amd64 --build-arg VLLM_VERSION=0.22.0 -t oicm/vllm:0.22.0 -f vLLM-podman/Dockerfile vLLM-podman
	@podman tag oicm/vllm:0.22.0 localhost/oicm/vllm:0.22.0-cu129


save-vllm-podman: # Save vLLM Podman image to tar file for transfer
	@podman save -o vllm-0.22.0-cu129.tar localhost/oicm/vllm:0.22.0-cu129

transfer-vllm-podman: # Transfer vLLM Podman image tar to remote server
	@rsync -vP --append-verify -e "ssh -c aes128-gcm@openssh.com" vllm-0.22.0-cu129.tar adeo@10.34.104.99:/home/adeo/vllm

build-sglang: # Build custom model server image for SGLang
	@docker build --platform=linux/amd64 -t oicm/sglang:0.5.9-cu129 -f sglang/Dockerfile SGLang

build-cputests: # Build CPU + Silero-VAD diagnostic image
	@podman build --platform=linux/amd64 -t cpu-vad-diag:latest -f cputests/Dockerfile cputests

build-diagnostics: # Build lightweight OICM diagnostic image
	@podman build --platform=linux/amd64 -t oicm/diagnostics:latest -f diagnostics/Dockerfile diagnostics


# podman save -o diagnostics-oicm.tar localhost/oicm/diagnostics:latest
# rsync -vP --append-verify -e "ssh -c aes128-gcm@openssh.com" diagnostics-oicm.tar adeo@10.34.104.99:/home/adeo