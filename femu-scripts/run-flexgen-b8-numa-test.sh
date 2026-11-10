#!/bin/bash
set -euo pipefail

# A/B changes only the inherited Host memory policy. Neither arm changes
# global sysctls, CPU affinity, mlock, pre-touch, or the aggregation operator.
if (( $# != 1 )) || [[ $1 != default && $1 != bind-all ]]; then
    echo 'Usage: run-flexgen-b8-numa-test.sh default|bind-all' >&2
    exit 1
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command -v numactl >/dev/null
if [[ $(cat /proc/sys/kernel/numa_balancing) != 1 ]]; then
    echo 'Expected Host kernel.numa_balancing=1 for this comparison; no settings were changed.' >&2
    exit 1
fi
export CEMU_NUMA_TEST_CASE=$1
export CEMU_NATIVE_NUMA_POLICY=$1
printf 'NUMA comparison arm: %s; no mlock/pre-touch changes.\n' "$1"
exec bash "$repo/femu-scripts/run-flexgen-b8-microbatch-test.sh"
