#!/bin/bash

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CEMU_CONFIG_FILE="${CEMU_CONFIG_FILE:-$script_dir/cemu_config_opt13b.json}"
export CEMU_NVM_BACKEND_DIR="${CEMU_NVM_BACKEND_DIR:-/data/lihan/cemu-nvm-opt13b}"
export CEMU_CSD_COUNT="${CEMU_CSD_COUNT:-1}"
export CEMU_GUEST_MEMORY="${CEMU_GUEST_MEMORY:-32G}"
export CEMU_COMPUTE_LOG="${CEMU_COMPUTE_LOG:-progress}"
export CEMU_COMPUTE_LOG_INTERVAL="${CEMU_COMPUTE_LOG_INTERVAL:-100}"

cd "$script_dir"
exec ./run-csd.sh "$@"
