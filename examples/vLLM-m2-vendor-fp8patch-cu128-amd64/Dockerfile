# HISTORICAL EXPERIMENT ONLY — DO NOT BUILD OR PUBLISH.
# Stock vendor passed the genuine FP8 factory / EP-locality regression.
# Live fix is COMPLETE sanitized quantization metadata, with NO source mount.
# This extends ONLY the exact independently audited vendor XLLM runtime.  The
# source used by the fork lives in /opt/vllm, not the generic OCI /vllm header.
ARG VENDOR_RUNTIME_IMAGE
FROM ${VENDOR_RUNTIME_IMAGE}

# Derived runtime image identity/labels; no changes to model, license or PVC.
LABEL org.opencontainers.image.title="m2-vllm-sparse-expert-fp8-r1" \
      org.opencontainers.image.description="Vendor M2/Xllm: compressed-tensors sparse expert weights/scales -> FusedMoE w13/w2 callback mapping"

ARG XLLM_PATCH_SHA256=1c820a27fd40727d1d220226ae82412341241767ebd54b015884f003d1489820
ARG XLLM_TEST_SHA256=553bcc887a385bf5525b7ee28fb16f24cbb31974e0146e6c56bfd84982798f5f
ARG VENDOR_RUNTIME_BASE=unknown
ENV AD_GATED_M2_PARENT="${VENDOR_RUNTIME_BASE}"

# Keep base ENTRYPOINT intact: private vLLM and existing gated public listener.
# This Docker file adds no hostNetworks, ports, caches, credentials, or PVCs.
COPY xllm_fp8_expert_mapping.patch /opt/m2-patches/
COPY test_xllm_mapping.py /opt/m2-patches/
RUN set -eux; \
    sha256sum -c <<<"${XLLM_PATCH_SHA256}  /opt/m2-patches/xllm_fp8_expert_mapping.patch"; \
    sha256sum -c <<<"${XLLM_TEST_SHA256}  /opt/m2-patches/test_xllm_mapping.py"; \
    patch --forward --dry-run -p0 -i /opt/m2-patches/xllm_fp8_expert_mapping.patch; \
    patch --forward -p0 -i /opt/m2-patches/xllm_fp8_expert_mapping.patch; \
    python3 -m py_compile /opt/vllm/vllm/model_executor/models/xllm.py; \
    python3 /opt/m2-patches/test_xllm_mapping.py; \
    test "$(sha256sum /opt/m2-patches/xllm_fp8_expert_mapping.patch | cut -d ' ' -f1)" = "${XLLM_PATCH_SHA256}"; \
    cp /opt/vllm/vllm/model_executor/models/xllm.py /opt/m2-patches/xllm.py.effective
