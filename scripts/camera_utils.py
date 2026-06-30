"""Resolve a V4L2 camera's /dev/videoN index by device NAME, not a fixed number.

/dev/videoN numbers shuffle on every replug/reboot (e.g. the C922 has been 8, then
14). Hardcoding an index breaks. find_camera_index("C922") reads the kernel device
name from sysfs and returns the node that can actually capture frames.
"""
from __future__ import annotations
import glob


def find_camera_index(name_substr: str, verify_capture: bool = True) -> int | None:
    """Return the /dev/videoN index whose sysfs name contains `name_substr` and
    that actually yields a frame. Returns None if not found.

    A USB cam exposes several nodes (capture + metadata); we pick the one that
    reads a frame. Only nodes matching `name_substr` are probed, so unrelated
    cameras (e.g. RealSense) are never opened.
    """
    cands = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            nm = open(path).read().strip()
        except OSError:
            continue
        if name_substr.lower() in nm.lower():
            cands.append(int(path.split("/")[-2].removeprefix("video")))
    cands.sort()
    if not cands or not verify_capture:
        return cands[0] if cands else None
    import cv2
    for idx in cands:
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return idx
    return cands[0]
