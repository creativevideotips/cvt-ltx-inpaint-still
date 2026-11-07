# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""The person check: finds people the model invented, with macOS's built-in detection.

person_check.swift does the finding, compiled once like the progress window. There are two ways a still
counts as having a person:

- At least half of a face or body detection lies in the fill. That ignores real people who stay in the
  picture, and still catches a new person where the mask had one, which comparing against the source
  would miss.
- Person segmentation marks a real area of the fill, more than a third of the person area kept outside
  it. This finds people seen from behind or out of focus, which the detections miss, as when a mask cuts
  through someone and the model finishes them. In an outpaint, the legs of a real person are small next
  to the rest of them (at most about a quarter in tests), so those pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import zlib

import numpy as np

IN_FILL = 0.5  # how much of a detection must lie in the fill
GRID_COLUMNS, GRID_ROWS = 192, 108  # also in person_check.swift
MIN_CELLS = 40  # person area in the fill, in grid cells, that counts: smaller is a speck
OF_KEPT = 1 / 3  # against the person area outside the fill
TIMEOUT_SECONDS = 60


def helper_path(cache_root: str) -> tuple[str, str]:
    """(path of the compiled helper, why not). Built on first use, one binary per source version."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person_check.swift")
    try:
        with open(src, "rb") as f:
            exe = os.path.join(cache_root, "bin", f"person_check-{hashlib.sha256(f.read()).hexdigest()[:12]}")
        if os.path.exists(exe):
            return exe, ""
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        tmp = f"{exe}.{os.getpid()}.tmp"  # two jobs may build at once; os.replace is atomic
        done = subprocess.run(["/usr/bin/xcrun", "swiftc", "-O", "-o", tmp, src], capture_output=True, text=True,
                              timeout=600)
        if done.returncode != 0:
            return "", "it could not be compiled (are the Command Line Tools installed?)"
        os.replace(tmp, exe)
        return exe, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"it could not be compiled ({type(exc).__name__})"


def write_png(path: str, picture: np.ndarray) -> None:
    """An 8-bit RGB PNG of a float picture, clipped to 0 to 1, in plain Python."""
    h, w = picture.shape[:2]
    pixels = (np.clip(picture[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    raw = np.zeros((h, w * 3 + 1), np.uint8)  # each row starts with filter type 0
    raw[:, 1:] = pixels.reshape(h, w * 3)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


def people_in_fill(boxes: list[dict], mask: np.ndarray) -> list[str]:
    """The kinds of detection ("face", "body", "upper") that mostly lie in the fill. Mask: (H,W), 1 in the fill."""
    h, w = mask.shape
    found = []
    for box in boxes:
        x0, y0 = max(0, int(box["x"] * w)), max(0, int(box["y"] * h))
        x1, y1 = min(w, int((box["x"] + box["w"]) * w)), min(h, int((box["y"] + box["h"]) * h))
        if x1 > x0 and y1 > y0 and float((mask[y0:y1, x0:x1] > 0.5).mean()) >= IN_FILL:
            found.append(box["kind"])
    return found


def person_area_in_fill(grid: str, mask: np.ndarray) -> bool:
    """True if segmentation marks a real area of the fill, large against the person area outside it."""
    if len(grid) != GRID_COLUMNS * GRID_ROWS:
        return False
    people = np.frombuffer(grid.encode(), np.uint8).reshape(GRID_ROWS, GRID_COLUMNS) == ord("1")
    h, w = mask.shape
    rows = np.arange(GRID_ROWS) * h // GRID_ROWS
    columns = np.arange(GRID_COLUMNS) * w // GRID_COLUMNS
    fill_cells = mask[np.ix_(rows, columns)] > 0.5
    inside = int((people & fill_cells).sum())
    outside = int((people & ~fill_cells).sum())
    return inside >= MIN_CELLS and inside > OF_KEPT * outside


def check(helper: str, plate: np.ndarray, mask: np.ndarray) -> tuple[list[str], str]:
    """(kinds of person found in the fill, why the check couldn't run). Never raises."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="ltxis_person_")
    os.close(fd)
    try:
        write_png(path, plate)
        done = subprocess.run([helper, path], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        if done.returncode != 0:
            return [], f"the detector stopped (exit {done.returncode})"
        result = json.loads(done.stdout.splitlines()[0])
        if "error" in result:
            return [], "the detector could not read the still"
        found = people_in_fill(result["boxes"], mask)
        if person_area_in_fill(result.get("grid", ""), mask):
            found.append("segmentation")
        return found, ""
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError) as exc:
        return [], f"the detector failed ({type(exc).__name__})"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
