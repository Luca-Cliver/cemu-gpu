#!/bin/bash

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${CEMU_BUILD_DIR:-$script_dir/../build}"

export CEMU_CONFIG_FILE="${CEMU_CONFIG_FILE:-$script_dir/cemu_config_instattention_figure14.json}"
export CEMU_NVM_BACKEND_DIR="${CEMU_NVM_BACKEND_DIR:-/data/lihan/cemu-nvm-figure14}"
export CEMU_CSD_COUNT="${CEMU_CSD_COUNT:-1}"
export CEMU_GUEST_MEMORY="${CEMU_GUEST_MEMORY:-64G}"
export CEMU_COMPUTE_LOG="${CEMU_COMPUTE_LOG:-progress}"
export CEMU_COMPUTE_LOG_INTERVAL="${CEMU_COMPUTE_LOG_INTERVAL:-100}"

cd "$build_dir"
exec ./run-csd.sh "$@"
