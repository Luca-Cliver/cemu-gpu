from pathlib import Path


def decode_mount_path(value):
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def require_fdmfs(directory):
    target = Path(directory).resolve()
    selected = None
    with open("/proc/self/mountinfo", encoding="utf-8") as mountinfo:
        for line in mountinfo:
            left, right = line.rstrip().split(" - ", 1)
            mountpoint = Path(decode_mount_path(left.split()[4]))
            try:
                target.relative_to(mountpoint)
            except ValueError:
                continue
            if selected is None or len(str(mountpoint)) > len(str(selected[0])):
                selected = (mountpoint, right.split()[0])
    if selected is None or selected[1] != "fdmfs":
        actual = "not mounted" if selected is None else f"{selected[1]} mounted at {selected[0]}"
        raise RuntimeError(
            f"--fdm-dir must reside on fdmfs: {target} ({actual}); "
            "run /root/cemu-mount.sh and verify with findmnt"
        )
