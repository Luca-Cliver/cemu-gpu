#!/usr/bin/env python3
"""Summarize the optional CEMU_SPARF_FTL_TRACE output."""

import argparse
import collections
import json
from pathlib import Path


def parse_record(line):
    fields = {}
    for item in line.split():
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value if key == "op" else int(value, 0)
    required = {"kind", "stream", "group", "page_in_group", "ch", "lun", "blk", "pg"}
    fields.setdefault("op", "write")
    return fields if required <= fields.keys() else None


def wait_summary(records):
    waits = [record.get("wait_ns", 0) for record in records]
    return {
        "total": sum(waits),
        "max": max(waits, default=0),
        "avg": sum(waits) / len(waits) if waits else 0,
    }


def summarize(path):
    records = []
    for line in path.read_text().splitlines():
        record = parse_record(line)
        if record is not None:
            records.append(record)
    writes = [record for record in records if record["op"] == "write"]
    reads = [record for record in records if record["op"] == "read"]
    kv_reads = [record for record in reads if record["kind"] in (0, 1, 2)]
    untagged_reads = [record for record in reads if record["kind"] not in (0, 1, 2)]
    identity_fields = {"controller", "nsid", "cookie"}
    missing_identity = sum(not identity_fields <= record.keys() for record in records)
    by_kind = collections.defaultdict(list)
    by_group = collections.defaultdict(list)
    physical = collections.Counter()
    for record in writes:
        by_kind[record["kind"]].append(record)
        device = (record.get("controller", "unknown"), record.get("nsid", "unknown"))
        group_key = (*device, record.get("cookie", "unknown"),
                     record["stream"], record["kind"], record["group"])
        by_group[group_key].append(record)
        physical[(*device, record["ch"], record["lun"], record.get("pl", 0),
                  record["blk"], record["pg"])] += 1
    groups = {}
    for key, items in by_group.items():
        groups["/".join(map(str, key))] = {
            "pages": len(items),
            "page_in_group": sorted(item["page_in_group"] for item in items),
            "channels": sorted({item["ch"] for item in items}),
            "luns": sorted({item["lun"] for item in items}),
        }
    return {
        "records": len(records),
        "writes": len(writes),
        "reads": len(reads),
        "kv_reads": len(kv_reads),
        "untagged_reads": len(untagged_reads),
        "records_without_identity": missing_identity,
        "read_wait_ns": wait_summary(reads),
        "kv_read_wait_ns": wait_summary(kv_reads),
        "untagged_read_wait_ns": wait_summary(untagged_reads),
        "kinds": {
            str(kind): {
                "pages": len(items),
                "channels": sorted({item["ch"] for item in items}),
                "luns": sorted({item["lun"] for item in items}),
            }
            for kind, items in sorted(by_kind.items())
        },
        "physical_pages_reused": sum(count - 1 for count in physical.values() if count > 1),
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.trace)
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n")
    print("[sparf-ftl] records=%d, writes=%d, reads=%d, groups=%d, physical_page_reuse=%d" % (
        summary["records"], summary["writes"], summary["reads"],
        len(summary["groups"]), summary["physical_pages_reused"]))
    if summary["reads"]:
        print("[sparf-ftl] all_read_wait_ns total=%d max=%d avg=%.1f" % (
            summary["read_wait_ns"]["total"], summary["read_wait_ns"]["max"],
            summary["read_wait_ns"]["avg"]))
        for label in ("kv", "untagged"):
            data = summary[label + "_read_wait_ns"]
            print("[sparf-ftl] %s_reads=%d wait_ns total=%d max=%d avg=%.1f" % (
                label, summary[label + "_reads"], data["total"], data["max"], data["avg"]))
    if summary["records_without_identity"]:
        print("[sparf-ftl] WARNING: %d legacy records lack full device/session identity; "
              "group and physical reuse counts may merge unrelated records" %
              summary["records_without_identity"])
    print("[sparf-ftl] Page wait sums are not end-to-end latency; physical reuse is not a GC error verdict.")
    for kind, data in summary["kinds"].items():
        print("[sparf-ftl] kind=%s pages=%d channels=%s luns=%s" % (
            kind, data["pages"], data["channels"], data["luns"]))


if __name__ == "__main__":
    main()
