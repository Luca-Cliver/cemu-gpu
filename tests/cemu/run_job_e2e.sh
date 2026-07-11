#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="root@localhost"
SSH_PORT="2222"
GUEST_DIR='$HOME/CEMU/tests/cemu'
MODE="native"
LOG_FILE=""
CSV_FILE=""
COMPUTE_CSV_FILE=""
COMPUTE_SUMMARY_FILE=""
SUMMARY_FILE=""
WARMUP_ITERS="1"

usage() {
    cat <<'EOF'
Usage:
  ./run_job_e2e.sh [options]
  ./run_job_e2e.sh [options] -- <custom cemu_benchmark args>

Options:
  --native              Run native CPU shared-library path (default).
  --gpu                 Run CUDA devptr path.
  --ssh TARGET          SSH target, default: root@localhost.
  --port PORT           SSH port, default: 2222.
  --guest-dir DIR       Guest benchmark dir, default: $HOME/CEMU/tests/cemu.
  --log FILE            Timestamped raw log output.
  --csv FILE            Per-job E2E CSV output.
  --compute-csv FILE    CEMU backend compute realtime CSV output.
  --compute-summary FILE
                        CEMU backend compute realtime summary output.
  --summary FILE        Summary output.
  --warmup N            Exclude iter < N from summary, default: 1.
  -h, --help            Show this help.

Default native command:
  ./build/cemu_benchmark -v -l ./build/kswitch_proxy.so -n kswitch_proxy -e 1.0 -o 0 -p 1 -c 16 -s 1 -d 0

Default GPU command:
  ./build/cemu_benchmark -v -u -l ./build/kswitch_proxy_devptr.so -n kswitch_proxy -e 1.0 -o 0 -p 1 -c 16 -s 1 -d 0
EOF
}

quote_args() {
    local out="" arg
    for arg in "$@"; do
        printf -v arg "%q" "$arg"
        out+="${arg} "
    done
    printf "%s" "${out% }"
}

CUSTOM_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --native)
        MODE="native"
        shift
        ;;
    --gpu)
        MODE="gpu"
        shift
        ;;
    --ssh)
        SSH_TARGET="$2"
        shift 2
        ;;
    --port)
        SSH_PORT="$2"
        shift 2
        ;;
    --guest-dir)
        GUEST_DIR="$2"
        shift 2
        ;;
    --log)
        LOG_FILE="$2"
        shift 2
        ;;
    --csv)
        CSV_FILE="$2"
        shift 2
        ;;
    --compute-csv)
        COMPUTE_CSV_FILE="$2"
        shift 2
        ;;
    --compute-summary)
        COMPUTE_SUMMARY_FILE="$2"
        shift 2
        ;;
    --summary)
        SUMMARY_FILE="$2"
        shift 2
        ;;
    --warmup)
        WARMUP_ITERS="$2"
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    --)
        shift
        CUSTOM_ARGS=("$@")
        break
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
done

if [[ "$MODE" == "gpu" ]]; then
    DEFAULT_ARGS=(./build/cemu_benchmark -v -u -l ./build/kswitch_proxy_devptr.so -n kswitch_proxy -e 1.0 -o 0 -p 1 -c 16 -s 1 -d 0)
else
    DEFAULT_ARGS=(./build/cemu_benchmark -v -l ./build/kswitch_proxy.so -n kswitch_proxy -e 1.0 -o 0 -p 1 -c 16 -s 1 -d 0)
fi

if [[ ${#CUSTOM_ARGS[@]} -gt 0 ]]; then
    BENCH_ARGS=("${CUSTOM_ARGS[@]}")
else
    BENCH_ARGS=("${DEFAULT_ARGS[@]}")
fi

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="e2e_${MODE}.log"
fi
if [[ -z "$CSV_FILE" ]]; then
    CSV_FILE="${LOG_FILE%.log}.csv"
fi
if [[ -z "$COMPUTE_CSV_FILE" ]]; then
    COMPUTE_CSV_FILE="${LOG_FILE%.log}_compute.csv"
fi
if [[ -z "$COMPUTE_SUMMARY_FILE" ]]; then
    COMPUTE_SUMMARY_FILE="${LOG_FILE%.log}_compute_summary.csv"
fi
if [[ -z "$SUMMARY_FILE" ]]; then
    SUMMARY_FILE="${LOG_FILE%.log}_summary.txt"
fi

BENCH_CMD=$(quote_args "${BENCH_ARGS[@]}")
REMOTE_CMD="cd ${GUEST_DIR} && stdbuf -oL ${BENCH_CMD}"

echo "SSH target: ${SSH_TARGET}:${SSH_PORT}"
echo "Guest dir:  ${GUEST_DIR}"
echo "Mode:       ${MODE}"
echo "Log:        ${LOG_FILE}"
echo "CSV:        ${CSV_FILE}"
echo "Compute CSV:${COMPUTE_CSV_FILE}"
echo "Compute Sum:${COMPUTE_SUMMARY_FILE}"
echo "Summary:    ${SUMMARY_FILE}"
echo "Command:    ${BENCH_CMD}"

ssh "${SSH_TARGET}" -p "${SSH_PORT}" "${REMOTE_CMD}" 2>&1 \
| while IFS= read -r line; do
    printf "%s %s\n" "$(date +%s%N)" "$line"
done | tee "${LOG_FILE}"

awk '
BEGIN {
    print "job,iter,e2e_ms";
}
/JOB_E2E_START/ {
    key = $3 " " $4;
    start[key] = $1;
}
/JOB_E2E_END/ {
    key = $3 " " $4;
    if (key in start) {
        split($3, job, "=");
        split($4, iter, "=");
        printf "%s,%s,%.3f\n", job[2], iter[2], ($1 - start[key]) / 1000000.0;
    }
}
' "${LOG_FILE}" > "${CSV_FILE}"

awk '
BEGIN {
    print "index,program,realtime_ms,runtime_ms";
}
/CEMU_COMPUTE:/ {
    program = "";
    realtime = "";
    runtime = "";
    for (i = 1; i <= NF; i++) {
        token = $i;
        gsub(/,/, "", token);
        if (token == "program" && i + 1 <= NF) {
            program = $(i + 1);
            gsub(/,/, "", program);
        } else if (token ~ /^realtime=/) {
            split(token, realtime_parts, "=");
            realtime = realtime_parts[2];
        } else if (token ~ /^runtime=/) {
            split(token, runtime_parts, "=");
            runtime = runtime_parts[2];
        }
    }
    if (realtime != "" && runtime != "") {
        printf "%d,%s,%.3f,%.3f\n",
               index++, program, realtime / 1000000.0, runtime / 1000000.0;
    }
}
' "${LOG_FILE}" > "${COMPUTE_CSV_FILE}"

awk -F, -v warmup="${WARMUP_ITERS}" '
BEGIN {
    print "metric,n,avg_ms,min_ms,max_ms,warmup_iters";
}
NR == 1 {
    next;
}
$1 >= warmup {
    realtime = $3;
    runtime = $4;
    rt_sum += realtime;
    sim_sum += runtime;
    n++;
    if (n == 1 || realtime < rt_min) {
        rt_min = realtime;
    }
    if (n == 1 || realtime > rt_max) {
        rt_max = realtime;
    }
    if (n == 1 || runtime < sim_min) {
        sim_min = runtime;
    }
    if (n == 1 || runtime > sim_max) {
        sim_max = runtime;
    }
}
END {
    if (n == 0) {
        printf "compute_realtime,0,nan,nan,nan,%s\n", warmup;
        printf "compute_runtime,0,nan,nan,nan,%s\n", warmup;
    } else {
        printf "compute_realtime,%d,%.3f,%.3f,%.3f,%s\n",
               n, rt_sum / n, rt_min, rt_max, warmup;
        printf "compute_runtime,%d,%.3f,%.3f,%.3f,%s\n",
               n, sim_sum / n, sim_min, sim_max, warmup;
    }
}
' "${COMPUTE_CSV_FILE}" > "${COMPUTE_SUMMARY_FILE}"

{
echo "e2e:"
awk -F, -v warmup="${WARMUP_ITERS}" '
NR == 1 {
    next;
}
$2 >= warmup {
    v = $3;
    sum += v;
    n++;
    if (n == 1 || v < min) {
        min = v;
    }
    if (n == 1 || v > max) {
        max = v;
    }
}
END {
    if (n == 0) {
        printf "n=0 avg_ms=nan min_ms=nan max_ms=nan warmup_iters=%s\n", warmup;
    } else {
        printf "n=%d avg_ms=%.3f min_ms=%.3f max_ms=%.3f warmup_iters=%s\n",
               n, sum / n, min, max, warmup;
    }
}
' "${CSV_FILE}"
echo "compute:"
cat "${COMPUTE_SUMMARY_FILE}"
} | tee "${SUMMARY_FILE}"
