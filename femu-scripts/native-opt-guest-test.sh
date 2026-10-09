#!/bin/bash
set -euo pipefail

if [[ $EUID != 0 ]] || ! grep -qw cemu_native_opt_smoke=1 /proc/cmdline; then
    echo 'Run only as root inside the Guest started by run-native-opt-test.sh.' >&2
    exit 1
fi
model=${1:-}
csd_target=cpu
read -r -a parameters < /proc/cmdline
for parameter in "${parameters[@]}"; do
    case "$parameter" in
        cemu_native_model=*) model=${model:-${parameter#cemu_native_model=}} ;;
        cemu_native_csd_target=*) csd_target=${parameter#cemu_native_csd_target=} ;;
    esac
done
case "$csd_target" in
    cpu) program=./build/sparf_attention.so ;;
    cuda) program=./build/sparf_attention_devptr.so ;;
    *) echo 'Invalid CSD target in Guest boot parameters.' >&2; exit 1 ;;
esac
model=${model:-OPT-125M}
case "$model" in
    OPT-125M) buffer_budget=67108864 ;;
    OPT-13B) buffer_budget=536870912 ;;
    *) echo 'Only OPT-125M and OPT-13B smoke tests are supported.' >&2; exit 1 ;;
esac
profile=${2:-smoke}
batch_size=1
prompt_length=32
decode_steps=4
write_batch_pages=32
runtime_args=()
case "$profile" in
    smoke) ;;
    b4-p1024-d16)
        if [[ "$model" != OPT-13B ]] || ! grep -qw cemu_native_profile=b4-p1024-d16 /proc/cmdline; then
            echo 'Restart with OPT-13B and CEMU_NATIVE_PROFILE=b4-p1024-d16 first.' >&2
            exit 1
        fi
        ratio=${3:-}
        if [[ ! -s "$ratio" ]]; then
            echo 'Pass a readable Softmax ratio JSON as argument 3; do not run with an empty ratio.' >&2
            exit 1
        fi
        ratio=$(readlink -f -- "$ratio")
        batch_size=4
        prompt_length=1024
        decode_steps=16
        write_batch_pages=256
        buffer_budget=1073741824
        runtime_args=(--runtime-config instattention/experiments/configs/opt13b_sparf_1csd.json
                      --softmax-ratio-file "$ratio")
        ;;
    *) echo 'Profile must be smoke or b4-p1024-d16.' >&2; exit 1 ;;
esac
mountpoint -q /sys || mount -t sysfs sysfs /sys
mountpoint -q /dev || mount -t devtmpfs devtmpfs /dev
mount -o remount,rw /
mkdir -p /root/models /root/cemu-logs /mnt/fdm0
mountpoint -q /root/models || mount -t 9p -o ro,trans=virtio,version=9p2000.L modelsrc /root/models
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
for path in /dev/ng0n1 /dev/ng0n3 /dev/nvme0c3; do
    if [[ ! -c "$path" ]]; then
        printf 'Missing native/CEMU character device: %s\n' "$path" >&2
        exit 1
    fi
done
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
source /root/cemu-gpu-env/bin/activate
python3 -c 'import torch; assert torch.cuda.is_available(), "Guest GPU unavailable"; print("Guest GPU:", torch.cuda.get_device_name(0))'

cd /root/CEMU/tests/cemu
if [[ ! -r "$program" ]]; then
    printf 'Missing CSD program: %s; build it on the Host first.\n' "$program" >&2
    exit 1
fi
if [[ "$profile" != smoke ]]; then
    PYTHONPATH=instattention python3 - "$ratio" <<'PY'
import sys
from experiments.config import load_experiment_config
from experiments.softmax_ratio import SoftmaxRatioTable
ratios = SoftmaxRatioTable.load(sys.argv[1])
ratios.validate_anchor(load_experiment_config("instattention/experiments/configs/opt13b_sparf_1csd.json").instcsd)
ratios.require_sparf(4, 40, 1024, 16, 8)
print("Softmax ratio coverage OK: microbatch=4, heads=40, prompt=1024, Decode=16")
PY
fi
make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
output=/root/cemu-logs/native-${model,,}-${profile}-${csd_target}-$(date +%Y%m%d-%H%M%S)
if [[ "$profile" != smoke ]]; then
    cp -- "$ratio" "$output-softmax.json"
    cp -- instattention/experiments/configs/opt13b_sparf_1csd.json "$output-runtime.json"
    runtime_args=(--runtime-config "$output-runtime.json" --softmax-ratio-file "$output-softmax.json")
fi
printf 'Guest log: %s.log\n' "$output"
printf 'CSD target from boot parameters: %s; program: %s\n' "$csd_target" "$program"
PYTHONPATH="/root/native-opt-client${PYTHONPATH:+:$PYTHONPATH}" \
    CEMU_INSTATTENTION_BUILD_DIR=/root/native-opt-client \
    python3 instattention/python/test_opt_cemu_pipeline.py \
    --model-dir "/root/models/$model" \
    --control /dev/nvme0c3 --namespace /dev/ng0n3 \
    --fdm-dir /mnt/fdm0 --csd-target "$csd_target" --program "$program" \
    --attention-mode sparf --sparf-native --sparf-autonomous --functional \
    --native-kv-device /dev/ng0n1 --native-kv-budget "$buffer_budget" \
    --batch-size "$batch_size" --gpu-batch-size "$batch_size" \
    --prompt-length "$prompt_length" --decode-steps "$decode_steps" \
    --attention-slots 2 --sparf-top-r 16 --sparf-compression-ratio 8 \
    --sparf-group-channels-per-page 8 --sparf-group-write-batch-pages "$write_batch_pages" \
    "${runtime_args[@]}" \
    --profile-output "$output-profile.csv" \
    2>&1 | tee "$output.log"
