SHELL :=/bin/bash

.DEFAULT_GOAL=help

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-vllm: # Build custom model server image for vLLM
	@docker build --platform=linux/arm64 -t oicm/vllm:0.19.0 -f vLLM/Dockerfile vLLM

build-sglang: # Build custom model server image for SGLang
	@docker build --platform=linux/arm64 -t oicm/sglang:0.5.9-cu129 -f sglang/Dockerfile SGLang