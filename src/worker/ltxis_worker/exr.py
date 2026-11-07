# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""OpenEXR files for the stills: half-float RGB scanlines, with the fill's matte in alpha.

The colour is the whole picture, not premultiplied by the alpha.

write_exr and read_exr are a small ZIP writer and reader, needing only Python's zlib. The worker writes
each still this way, in the cache. write_dwab turns it into the one file that is kept, with the official
OpenEXR package; if that isn't possible, the ZIP is what gets saved. The reader also takes uncompressed
files, which is what the first builds wrote.
String metadata goes in the header, where Fusion and other tools can read it.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np

MAGIC = 0x01312F76
VERSION = 2
PIXEL_HALF = 1
CHANNELS = (("B", 2), ("G", 1), ("R", 0))  # EXR stores channels sorted by name
WITH_ALPHA = (("A", 3),) + CHANNELS
COMPRESSION_NONE, COMPRESSION_ZIP = 0, 3
ZIP_LINES = 16  # EXR's ZIP keeps 16 scanlines in a block


def _attribute(name: str, kind: str, data: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(data)) + data


def _header(width: int, height: int, metadata: dict[str, str], layout=CHANNELS) -> bytes:  # noqa: ANN001
    channels = b"".join(name.encode() + b"\0" + struct.pack("<iB3xii", PIXEL_HALF, 0, 1, 1) for name, _ in layout)
    window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    parts = [
        _attribute("channels", "chlist", channels + b"\0"),
        _attribute("compression", "compression", bytes([COMPRESSION_ZIP])),
        _attribute("dataWindow", "box2i", window),
        _attribute("displayWindow", "box2i", window),
        _attribute("lineOrder", "lineOrder", b"\0"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]
    parts += [_attribute(key, "string", value.encode("utf-8")) for key, value in sorted(metadata.items())]
    return b"".join(parts) + b"\0"


def _pack(raw: bytes) -> bytes:
    """EXR's ZIP: split into two half-size runs of bytes, store each byte as its difference from the
    one before, then deflate. A block that would not shrink is stored as it is, as the format says."""
    data = np.frombuffer(raw, dtype=np.uint8)
    half = (data.size + 1) // 2
    split = np.empty_like(data)
    split[:half], split[half:] = data[::2], data[1::2]
    predicted = split.copy()
    predicted[1:] = (split[1:].astype(np.int16) - split[:-1].astype(np.int16) + 128 + 256).astype(np.uint8)
    packed = zlib.compress(predicted.tobytes(), 6)
    return packed if len(packed) < len(raw) else raw


def _unpack(packed: bytes, raw_size: int) -> bytes:
    if len(packed) == raw_size:
        return packed
    predicted = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    if predicted.size != raw_size:
        raise ValueError("a block has the wrong size")
    # Undo the differences (bytes wrap around, so the running sum wraps too).
    split = (np.cumsum(predicted.astype(np.int64) - 128) + 128).astype(np.uint8) if predicted.size else predicted
    half = (raw_size + 1) // 2
    data = np.empty_like(split)
    data[::2], data[1::2] = split[:half], split[half:]
    return data.tobytes()


def write_exr(path: str, rgb: np.ndarray, metadata: dict[str, str] | None = None,
              alpha: np.ndarray | None = None) -> None:
    """Write (H,W,3) float values, and (H,W) alpha if given, to `path`, replacing it only once the file is
    complete."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3), got {rgb.shape}")
    height, width, _ = rgb.shape
    pixels = rgb if alpha is None else np.dstack([rgb, alpha])
    layout = CHANNELS if alpha is None else WITH_ALPHA
    half = np.ascontiguousarray(pixels, dtype="<f2")
    header = struct.pack("<ii", MAGIC, VERSION) + _header(width, height, metadata or {}, layout)
    blocks = []
    for y in range(0, height, ZIP_LINES):
        lines = half[y:y + ZIP_LINES]
        raw = b"".join(lines[i, :, c].tobytes() for i in range(lines.shape[0]) for _, c in layout)
        blocks.append((y, _pack(raw)))
    position = len(header) + 8 * len(blocks)
    offsets = []
    for _, packed in blocks:
        offsets.append(position)
        position += 8 + len(packed)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        for y, packed in blocks:
            f.write(struct.pack("<ii", y, len(packed)))
            f.write(packed)
    os.replace(tmp, path)


OPENEXR_REQUIREMENT = "openexr==3.5.3"


def load_openexr(library_dir: str, uv: str):  # noqa: ANN201 - the OpenEXR module, or None and the reason
    """The official OpenEXR package, for DWAB: writing that by hand is not worth it. It goes in this
    effect's own folder, not the runtime it shares with Alpha Gen, installed on first use."""
    import subprocess
    import sys

    if library_dir not in sys.path:
        sys.path.insert(0, library_dir)
    try:
        import OpenEXR

        return OpenEXR, ""
    except ImportError:
        pass
    if not uv or not os.path.exists(uv):
        return None, "the installer tool was not found"
    try:
        done = subprocess.run([uv, "pip", "install", "--quiet", "--python", sys.executable, "--target", library_dir,
                               "--no-deps", OPENEXR_REQUIREMENT], capture_output=True, text=True, timeout=180,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"installing it failed ({type(exc).__name__})"
    if done.returncode != 0:
        return None, "installing it failed: " + (done.stderr.strip().splitlines() or ["no message"])[-1][:120]
    import importlib

    importlib.invalidate_caches()
    try:
        import OpenEXR

        return OpenEXR, ""
    except ImportError as exc:
        return None, f"it would not load ({exc})"


def write_dwab(openexr, path: str, rgb: np.ndarray, level: float, metadata: dict[str, str] | None = None,  # noqa: ANN001
               alpha: np.ndarray | None = None) -> None:
    """DWAB: lossy for colour and much smaller than ZIP; a channel named A stays lossless. Replaces `path`
    only once the file is complete."""
    header = {"compression": openexr.DWAB_COMPRESSION, "dwaCompressionLevel": float(level),
              "type": openexr.scanlineimage, **(metadata or {})}
    channels = {name: rgb[..., c].astype(np.float16) for name, c in CHANNELS}
    if alpha is not None:
        channels["A"] = alpha.astype(np.float16)
    tmp = f"{path}.tmp{os.getpid()}.exr"
    try:
        with openexr.File(header, channels) as f:
            f.write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_exr(path: str) -> tuple[np.ndarray, dict[str, str]]:
    """Read back a file from write_exr: (H,W,3) float32 and its string metadata. For tests and tools."""
    rgb, _alpha, metadata = read_exr_with_alpha(path)
    return rgb, metadata


def read_exr_with_alpha(path: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, str]]:
    """(H,W,3) float32, (H,W) alpha or None if it has none, and its string metadata."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != MAGIC or version & 0xFF != VERSION:
        raise ValueError(f"not an EXR this reader knows: {path}")
    pos, width, height, metadata, compression = 8, 0, 0, {}, -1
    layout = CHANNELS
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode()
        pos = end + 1
        end = data.index(b"\0", pos)
        kind = data[pos:end].decode()
        pos = end + 1
        size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if name == "dataWindow":
            x0, y0, x1, y1 = struct.unpack_from("<iiii", data, pos)
            width, height = x1 - x0 + 1, y1 - y0 + 1
        elif name == "compression":
            compression = data[pos]
        elif name == "channels":
            layout = WITH_ALPHA if data[pos:pos + 2] == b"A\0" else CHANNELS
        elif kind == "string":
            metadata[name] = data[pos:pos + size].decode("utf-8")
        pos += size
    pos += 1
    if compression not in (COMPRESSION_NONE, COMPRESSION_ZIP):
        raise ValueError(f"this reader only handles files this plugin wrote: {path}")
    lines_per_block = ZIP_LINES if compression == COMPRESSION_ZIP else 1
    count = (height + lines_per_block - 1) // lines_per_block
    offsets = np.frombuffer(data, dtype="<u8", count=count, offset=pos)
    out = np.empty((height, width, len(layout)), dtype=np.float32)
    row_bytes = width * 2
    for at in offsets:
        y, size = struct.unpack_from("<ii", data, int(at))
        lines = min(lines_per_block, height - y)
        raw_size = lines * row_bytes * len(layout)
        raw = _unpack(data[int(at) + 8:int(at) + 8 + size], raw_size)
        values = np.frombuffer(raw, dtype="<f2")
        for line in range(lines):
            for i, (_, c) in enumerate(layout):
                start = (line * len(layout) + i) * width
                out[y + line, :, c] = values[start:start + width]
    return out[..., :3], (out[..., 3] if len(layout) == 4 else None), metadata
