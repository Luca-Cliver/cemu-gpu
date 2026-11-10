"""Build two aggregation modules identically and compare only operator calls.

Usage: python3 bench_kv_aggregate.py --baseline saved-before.c --output /path/to/new-dir
The baseline must be a snapshot of the working file, not an unrelated git HEAD.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/cemu/instattention/python'))
from test_instattention_qemu_io import InstattentionQemuIoTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=11)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--decode-cache-mib', type=int, default=0,
                        help='Scan this many MiB before each Decode call, outside timing (0..256).')
    parser.add_argument('--interleaved-batches', type=int, choices=(0, 4, 8), default=0,
                        help='Synthetic Decode window across 40 layers and B4/B8, without historical KV.')
    parser.add_argument('--page-ab', action='store_true',
                        help='Use baseline operator for BOTH variants; compare MADV_NOHUGEPAGE vs MADV_HUGEPAGE.')
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('repeats must be between 3 and 100')
    if not 0 <= args.decode_cache_mib <= 256:
        parser.error('decode-cache-mib must be between 0 and 256')
    if args.page_ab and not args.interleaved_batches:
        parser.error('page-ab requires interleaved-batches')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    flags = ['-std=c11', '-O2', '-march=native', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC']
    sources = [args.baseline.resolve(), ROOT / 'tests/cemu/kernel/fdm_kv_aggregate.c']
    libraries = []
    hashes = {}
    for name, source in zip(('before', 'after'), sources):
        raw = source.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        (directory / (name + '.c')).write_bytes(raw)
        # The snapshot keeps its original relative includes. Resolve those as
        # if it were still located in tests/cemu/kernel.
        library = directory / (name + '.so')
        subprocess.run(['cc', *flags, '-I', str(ROOT / 'tests/cemu/kernel'),
                        str(directory / (name + '.c')), str(ROOT / 'hw/femu/fdm_kv_layout.c'),
                        '-o', str(library)], check=True)
        libraries.append(library)
    harness = InstattentionQemuIoTest()
    executable = harness.build_sources([
        'hw/femu/fdm_kv_layout.c', 'hw/femu/fdm_kv_buffer.c', 'hw/femu/fdm_kv_receive.c',
        'tests/cemu/instattention/experiments/bench_kv_aggregate.c'], str(directory))
    available = os.sched_getaffinity(0)
    cpu = min(available) if args.cpu is None else args.cpu
    if cpu not in available:
        parser.error('requested CPU is outside allowed affinity')
    os.sched_setaffinity(0, {cpu})
    command = [str(executable), *(str(p) for p in libraries), str(args.repeats), str(args.decode_cache_mib),
               str(args.interleaved_batches), str(int(args.page_ab))]
    result = subprocess.run(command, text=True, capture_output=True)
    (directory / 'samples.csv').write_text(result.stdout)
    (directory / 'validation.txt').write_text(result.stderr)
    result.check_returncode()
    huge_pages = [dict(variant=int(variant), data_bytes=int(size), anon_huge_kib=int(kib))
                  for variant, size, kib in re.findall(
                      r'page-ab variant=(\d+) data_bytes=(\d+) AnonHugePages_KiB=(\d+)', result.stderr)]
    if args.page_ab and not any(row['variant'] == 1 and row['anon_huge_kib'] for row in huge_pages):
        print('WARNING: no actual huge pages observed; this run cannot establish a huge-page performance benefit.')
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    groups = {}
    for row in rows:
        if int(row['round']) == 0:
            continue
        key = (int(row['element_bytes']), row['phase'])
        groups.setdefault(key, {0: [], 1: []})[int(row['variant'])].append(
            int(row['total_ns']) / int(row['calls']) / 1000)
    summary = []
    for (element, phase), variants in groups.items():
        before, after = (statistics.median(variants[index]) for index in (0, 1))
        entry = dict(element_bytes=element, phase=phase, before_us_per_call=before,
                     after_us_per_call=after, speedup=before / after,
                     reduction_percent=100 * (1 - after / before),
                     before_min_max_us=[min(variants[0]), max(variants[0])],
                     after_min_max_us=[min(variants[1]), max(variants[1])])
        summary.append(entry)
        print(f'FP{element*8} {phase}: {before:.3f} -> {after:.3f} us/call; '
              f'{before/after:.2f}x, reduction={entry["reduction_percent"]:.1f}%')
    (directory / 'summary.json').write_text(json.dumps(dict(
        cpu=cpu, platform=platform.platform(), compiler=subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
        cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')), 'unknown'),
        module_flags=flags, source_sha256=hashes, repeats=args.repeats, results=summary,
        decode_cache_perturbation_mib=args.decode_cache_mib,
        interleaved_batches=args.interleaved_batches,
        observed_page_backing=huge_pages,
        comparison=('Same baseline operator: MADV_NOHUGEPAGE vs MADV_HUGEPAGE, discard and pre-touch outside timing'
                    if args.page_ab else 'Baseline operator vs optimized operator'),
        workload=('Synthetic positions 0..15, 40 layers, all batches interleaved; no historical KV; '
                  'same group offsets as P1024+D16. FP32 also allocates token pages at step 8.'
                  if args.interleaved_batches else 'Single active batch/layer with P1024 history and D16'),
        methodology='Single CPU; alternate A/B order; round 0 discarded; normal unprofiled entry; '
        'H40,d128,4KiB pages,8 channels/page,128KiB input; see workload for stream order; '
        'no NAND writer; pre-touched memory; packet preparation, initialization and '
        'validation excluded; optional cache-buffer scan before Decode excluded; synthetic disturbance, '
        'not actual interleaved layers or a guarantee of cold caches. Full metadata/data byte comparison '
        'each pair. Host time only, not E2E.'
    ), indent=2) + '\n')
    print(result.stderr.strip())
    print('Saved:', directory)


if __name__ == '__main__':
    main()
