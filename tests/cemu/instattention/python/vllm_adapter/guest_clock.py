"""Select the same freeze-aware Guest clock as the existing OPT experiments."""

from pathlib import Path
import time


CLOCK_DIRECTORY = Path("/sys/devices/system/clocksource/clocksource0")


def configure_guest_clock(*, cmdline=Path("/proc/cmdline"), directory=CLOCK_DIRECTORY):
    # Never change the Host clocksource, even if run.py is invoked there by mistake.
    if "cemu_native_opt_smoke=1" not in cmdline.read_text().split():
        raise RuntimeError("run this entry point inside the CEMU Guest; refusing to change Host clocksource")
    current = directory / "current_clocksource"
    before = current.read_text().strip()
    if "hpet" not in (directory / "available_clocksource").read_text().split():
        raise RuntimeError("Guest HPET unavailable; cannot report freeze-excluded timings")
    if before != "hpet":
        current.write_text("hpet\n")
    verify_guest_clock(directory=directory)
    info = time.get_clock_info("perf_counter")
    return {
        "before": before, "selected": "hpet", "perf_counter": info.implementation,
        "resolution_seconds": info.resolution,
        "freeze_policy": "Guest HPET / QEMU virtual time exclude functional freeze; no manual subtraction",
        "cuda_events": "GPU clock is not frozen with Guest; diagnostic only, not additive to HPET time",
    }


def verify_guest_clock(*, directory=CLOCK_DIRECTORY):
    actual = (directory / "current_clocksource").read_text().strip()
    if actual != "hpet":
        raise RuntimeError(f"Guest clocksource changed to {actual}; discard timing results (requires hpet)")
