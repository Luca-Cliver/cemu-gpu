"""Read/queue diagnostics on QEMU's virtual clock, independent of host timing."""
from collections import defaultdict
from statistics import median


def command_key(row):
    return tuple(row[k] for k in ("pid", "session", "controller", "command"))


def nonnegative(row, fields):
    for name in fields:
        value = row[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid {name}")


def validate_completion(row):
    if row["schema"] != 1 or row["clock"] != "qemu_virtual_ns":
        raise ValueError("unsupported completion schema/clock")
    nonnegative(row, ("pid", "session", "controller", "command", "received_ns",
                      "start_ns", "finish_ns", "scheduled_runtime_ns", "cu"))
    if not row["received_ns"] <= row["start_ns"] <= row["finish_ns"]:
        raise ValueError("invalid scheduler timestamps")


def validate_diagnostics(row):
    if "virtual" not in row:
        if "reads" in row:
            raise ValueError("reads without a virtual clock")
        return
    virtual = row["virtual"]
    if virtual["clock"] != "qemu_virtual_ns":
        raise ValueError("unsupported virtual clock")
    nonnegative(virtual, ("functional_start_ns", "functional_end_ns"))
    if virtual["functional_end_ns"] < virtual["functional_start_ns"]:
        raise ValueError("virtual functional time went backwards")
    names = set()
    for read in row["reads"]:
        name = read["name"]
        if name not in ("channel_k", "token_k", "token_v") or name in names:
            raise ValueError("invalid/duplicate read phase")
        names.add(name)
        nonnegative(read, ("logical_bytes", "submitted_bytes", "ranges", "model_start_ns",
            "model_end_ns", "pages", "unmapped_pages", "service_ns", "queue_ns",
            "max_queue_ns", "critical_queue_ns", "critical_service_ns",
            "critical_initial_wait_ns", "channels", "luns_per_channel", "page_bytes", "page_read_ns"))
        if read["model_end_ns"] < read["model_start_ns"]:
            raise ValueError("invalid modeled read interval")
        counts = read["lun_pages"]
        if (len(counts) != read["channels"] * read["luns_per_channel"] or
                any(type(n) is not int or n < 0 for n in counts) or sum(counts) != read["pages"]):
            raise ValueError("inconsistent LUN page counts")
        if read["pages"] and (not read["page_bytes"] or not read["page_read_ns"]):
            raise ValueError("missing NAND geometry")
        if read["critical_initial_wait_ns"] > read["critical_queue_ns"]:
            raise ValueError("initial wait exceeds critical queue time")


def join_completions(records, completions):
    by_command = {}
    for row in records:
        key = command_key(row)
        if key in by_command:
            raise ValueError("duplicate command identity")
        by_command[key] = row
    for event in completions:
        row = by_command.get(command_key(event))
        if row is None:
            raise ValueError("scheduler completion without a functional record")
        if "scheduler" in row:
            raise ValueError("duplicate scheduler completion")
        row["scheduler"] = event


def summarize_model(rows):
    observed = [r for r in rows if "virtual" in r]
    if not observed:
        return {"available": False}
    successful = [r for r in observed if r["success"]]
    finished = [r for r in successful if "scheduler" in r]
    phases = defaultdict(list)
    for row in successful:
        for read in row["reads"]:
            phases[read["name"]].append(read)
    reads = {}
    for name, items in sorted(phases.items()):
        totals = {key: sum(r[key] for r in items) for key in (
            "logical_bytes", "submitted_bytes", "pages", "unmapped_pages", "service_ns",
            "queue_ns", "critical_queue_ns", "critical_service_ns", "critical_initial_wait_ns")}
        totals["requests"] = len(items)
        totals["model_ns"] = sum(r["model_end_ns"] - r["model_start_ns"] for r in items)
        totals["nand_bytes"] = sum(r["pages"] * r["page_bytes"] for r in items)
        totals["page_amplification"] = (totals["nand_bytes"] / totals["logical_bytes"]
                                        if totals["logical_bytes"] else None)
        totals["active_luns_mean"] = sum(sum(n > 0 for n in r["lun_pages"]) for r in items) / len(items)
        imbalances = [max(r["lun_pages"]) * len(r["lun_pages"]) / r["pages"]
                      for r in items if r["pages"]]
        totals["lun_max_to_mean_median"] = median(imbalances) if imbalances else None
        # Ideal full-page throughput of this LUN timing model, NOT useful KV bandwidth.
        totals["ideal_page_gbps"] = sorted({r["channels"] * r["luns_per_channel"] *
            r["page_bytes"] / r["page_read_ns"] for r in items if r["page_read_ns"]})
        totals["geometry"] = [list(g) for g in sorted({(r["channels"], r["luns_per_channel"],
                              r["page_bytes"], r["page_read_ns"]) for r in items})]
        reads[name] = totals

    workflows = defaultdict(list)
    for r in successful:
        key = (r["program"], r["workflow"] if r["stage"] else r["command"])
        workflows[key].append(r)
    layer_groups = defaultdict(list)
    boundary_gaps = []
    for commands in workflows.values():
        ordered = sorted(commands, key=lambda r: r["stage"])
        expected = [1, 2, 3, 4] if ordered[0]["stage"] else [0]
        if [r["stage"] for r in ordered] != expected or any("scheduler" not in r for r in ordered):
            continue
        times = [r["scheduler"] for r in ordered]
        boundary_gaps.extend(b["received_ns"] - a["finish_ns"] for a, b in zip(times, times[1:]))
        layer_groups[(ordered[0]["layer"], ordered[0]["tokens"])].append(
            (times[0]["received_ns"], times[-1]["finish_ns"]))
    submission_gaps = []
    for spans in layer_groups.values():
        spans.sort()
        # Signed gap: negative means the next request arrived before the previous
        # one completed; it does not prove independent NAND/CU service overlap.
        submission_gaps.extend(b[0] - a[1] for a, b in zip(spans, spans[1:]))
    scheduler = {
        "completed_commands": len(finished),
        "missing_completions": len(observed) - sum("scheduler" in r for r in observed),
        "queue_sum_ns": sum(r["scheduler"]["start_ns"] - r["scheduler"]["received_ns"] for r in finished),
        "job_span_sum_ns": sum(r["scheduler"]["finish_ns"] - r["scheduler"]["start_ns"] for r in finished),
        "charged_runtime_sum_ns": sum(r["scheduler"]["scheduled_runtime_ns"] for r in finished),
        "stage_boundary_gap_sum_ns": sum(boundary_gaps),
        "stage_boundary_negative_gaps": sum(g < 0 for g in boundary_gaps),
        "microbatch_pairs": len(submission_gaps),
        "next_received_before_previous_finish": sum(g < 0 for g in submission_gaps),
        "submission_gap_median_ns": median(submission_gaps) if submission_gaps else None,
    }
    return dict(available=True, observed_commands=len(observed), reads=reads, scheduler=scheduler,
        virtual_functional_advance_sum_ns=sum(r["virtual"]["functional_end_ns"] -
                                            r["virtual"]["functional_start_ns"] for r in observed))


def print_model(report):
    if not report["available"]:
        print("  virtual/read diagnostics: unavailable (old log; not zero)")
        return
    for name, r in report["reads"].items():
        amplification = r["page_amplification"]
        print(f"  NAND {name}: model={r['model_ns']/1e9:.6f}s, pages={r['pages']}, "
              f"unmapped={r['unmapped_pages']}, page/useful={amplification}, "
              f"active LUNs(mean)={r['active_luns_mean']:.1f}, "
              f"LUN max/mean(median)={r['lun_max_to_mean_median']}")
        print(f"    critical-page wait: initial={r['critical_initial_wait_ns']/1e9:.6f}s, "
              f"remaining={(r['critical_queue_ns']-r['critical_initial_wait_ns'])/1e9:.6f}s; "
              f"ideal full-page capacity={r['ideal_page_gbps']} GB/s")
    s = report["scheduler"]
    print(f"  virtual scheduler: completed={s['completed_commands']}, missing={s['missing_completions']}, "
              f"received-to-start(sum, includes dispatch)={s['queue_sum_ns']/1e9:.6f}s, "
              f"stage gaps(sum)={s['stage_boundary_gap_sum_ns']/1e9:.6f}s")
    print(f"  next microbatch received before previous completion: "
          f"{s['next_received_before_previous_finish']}/{s['microbatch_pairs']}; "
          f"signed submission gap(median)={s['submission_gap_median_ns']} ns")
