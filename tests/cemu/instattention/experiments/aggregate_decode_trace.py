"""Split existing device traces by Decode position and page-allocation path.

Host elapsed times include instrumentation/preemption; these are not CPU times
or simulated runtimes. No rerun or new device instrumentation is required.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def summarize(rows):
    elapsed = sorted(row['aggregate_host_ns'] for row in rows)
    stages = {key: sum(row['aggregate_profile']['host_ns'][key] for row in rows)
              for key in rows[0]['aggregate_profile']['host_ns']}
    result = dict(
        requests=len(rows), calls=sum(row['aggregate_profile']['calls'] for row in rows),
        new_pages=sum(row['aggregate_profile']['new_pages'] for row in rows),
        copied_bytes=sum(row['aggregate_profile']['copied_bytes'] for row in rows),
        host_total_seconds=sum(elapsed) / 1e9,
        host_mean_us=statistics.mean(elapsed) / 1000,
        host_p50_us=statistics.median(elapsed) / 1000,
        host_p99_us=elapsed[math.ceil(len(elapsed) * .99) - 1] / 1000,
        host_max_us=elapsed[-1] / 1000,
        stage_total_seconds={key: value / 1e9 for key, value in stages.items()},
        stage_mean_us={key: value / len(rows) / 1000 for key, value in stages.items()})
    if all(row.get('aggregate_probe_calls', 0) == row['aggregate_profile']['calls'] for row in rows):
        result['execution_probe'] = {key: sum(row['aggregate_' + key] for row in rows) for key in
                                     ('probe_calls', 'thread_cpu_ns', 'minor_faults', 'major_faults',
                                      'voluntary_switches', 'involuntary_switches')}
        result['execution_probe']['note'] = 'Probe brackets are slightly wider than Host timer; differences are approximate, not exact off-CPU time.'
    if all('aggregate_location' in row for row in rows):
        result['cpu_end_counts'] = dict(Counter(str(row['aggregate_location']['cpu_end']) for row in rows))
        result['observed_cpu_changes'] = sum(row['aggregate_location']['cpu_changes'] for row in rows)
    return result


def analyze(path, decode_start, require_probe=False):
    by_session = defaultdict(list)
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if row.get('source') != 'device' or row.get('operation') != 1 or not row.get('aggregate_profile'):
                continue
            by_session[(row['pid'], row['controller'], row['session'])].append(row)
    if len(by_session) != 1:
        raise ValueError(f'Expected exactly one profiled KV session, found {len(by_session)}; use a per-run trace.')
    rows = next(iter(by_session.values()))
    paths, tokens = defaultdict(list), defaultdict(list)
    existing_cpu, existing_speed = defaultdict(list), defaultdict(list)
    slow_requests = []
    for row in rows:
        if row['token_first'] < decode_start:
            continue
        profile = row['aggregate_profile']
        if profile['version'] != 1:
            raise ValueError('Unsupported aggregation profile version')
        if sum(profile['host_ns'].values()) > row['aggregate_host_ns']:
            raise ValueError('Internal stages exceed Host elapsed time')
        kind = ('retry_or_error' if profile['calls'] != 1 or profile['successful_calls'] != 1 or row['status']
                else 'allocating' if profile['new_pages'] else 'existing')
        paths[kind].append(row)
        tokens[row['token_first']].append(row)
        if require_probe and (row.get('aggregate_probe_calls', 0) != profile['calls'] or
                              'aggregate_location' not in row):
            raise ValueError('Execution/location probes missing: restart with the rebuilt QEMU and detailed aggregate profile.')
        if kind == 'existing':
            slow = row['aggregate_host_ns'] >= 250000
            existing_speed['at_least_250us' if slow else 'below_250us'].append(row)
            if 'aggregate_location' in row:
                existing_cpu[str(row['aggregate_location']['cpu_end'])].append(row)
            if slow:
                slow_requests.append({key: row[key] for key in row if key in (
                    'packet_id', 'token_first', 'qid', 'aggregate_host_ns', 'aggregate_profile', 'aggregate_location')
                    or key.startswith('aggregate_') and key not in ('aggregate_ns',)})
    if not tokens:
        raise ValueError('No profiled Decode requests')
    return dict(source=str(path), decode_start=decode_start,
                scope='Host elapsed per request, including profile overhead and possible preemption; not CPU or model time.',
                paths={key: summarize(value) for key, value in paths.items()},
                tokens={str(key): summarize(value) for key, value in sorted(tokens.items())},
                existing_by_cpu={key: summarize(value) for key, value in existing_cpu.items()},
                existing_by_duration={key: summarize(value) for key, value in existing_speed.items()},
                slow_existing_requests=slow_requests,
                slow_threshold_us=250)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--decode-start', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-probe', action='store_true')
    args = parser.parse_args()
    report = analyze(args.trace, args.decode_start, args.require_probe)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    def probe_text(value):
        probe = value.get('execution_probe')
        if not probe:
            return '; CPU/fault/switch probe unavailable'
        return (f'; cpu_avg={probe["thread_cpu_ns"] / value["requests"] / 1000:.3f}us'
                f', minor_faults={probe["minor_faults"]}, major_faults={probe["major_faults"]}'
                f', voluntary_switches={probe["voluntary_switches"]}'
                f', involuntary_switches={probe["involuntary_switches"]}')
    for key, value in report['paths'].items():
        print(f'[aggregate-decode] {key}: n={value["requests"]}, total={value["host_total_seconds"]:.6f}s, '
              f'avg={value["host_mean_us"]:.3f}us, p50={value["host_p50_us"]:.3f}us, '
              f'p99={value["host_p99_us"]:.3f}us, new_pages={value["new_pages"]}' + probe_text(value))
    for key, value in report['tokens'].items():
        print(f'[aggregate-decode] token={key}: avg={value["host_mean_us"]:.3f}us, p50={value["host_p50_us"]:.3f}us, '
              f'channel_k={value["stage_mean_us"]["channel_k"]:.3f}us' + probe_text(value))
    for category in ('existing_by_cpu', 'existing_by_duration'):
        for key, value in report[category].items():
            print(f'[aggregate-decode] {category}={key}: n={value["requests"]}, '
                  f'avg={value["host_mean_us"]:.3f}us' + probe_text(value))


if __name__ == '__main__':
    main()
