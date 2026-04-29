#!/bin/bash
if [[ "${INIT_CONTAINER_USE_PVC}" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/pvc-home"}
    echo "INIT_CONTAINER_USE_PVC is set. Using PVC path: ${BASE_PATH}"
elif [[ "$USE_DATA_VOLUME" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/data-volume"}
    echo "USE_DATA_VOLUME is set. Using DATA VOLUME path: ${BASE_PATH}"
else
    BASE_PATH="/tmp/oicm"
    mkdir -p "${BASE_PATH}" || { echo "Failed to create directory: ${BASE_PATH}"; exit 1; }
    echo "INIT_CONTAINER_USE_PVC is not set. Using temporary path: ${BASE_PATH}"
fi
export BASE_PATH

export BASE_DOWNLOAD_FOLDER="${BASE_PATH}/app/download"

if [[ "$USE_DATA_VOLUME" == "True" ]]; then
    MODEL_DOWNLOAD_FOLDER="${BASE_PATH}"
else
    DEFAULT_MODEL_DOWNLOAD_FOLDER="${BASE_DOWNLOAD_FOLDER}/base_model"
    MODEL_DOWNLOAD_FOLDER="${MODEL_DOWNLOAD_FOLDER:-${DEFAULT_MODEL_DOWNLOAD_FOLDER}}"
fi


python3 -m sglang.launch_server \
    --host=0.0.0.0 --port=8080 \
    --model-path="$MODEL_DOWNLOAD_FOLDER" \
    --served-model-name="$MODEL_ID" \
    --enable-metrics