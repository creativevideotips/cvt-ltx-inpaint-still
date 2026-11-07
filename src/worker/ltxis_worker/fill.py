# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""The recipe that makes one clean plate still from a picture and a mask.

Two methods, from the tests on four shots (cvtLTXInpaintTest/ltx_inpaint_test.py):
  detail     one IC-LoRA stage at the working size (8 steps). The most detail. Can invent objects
             in a very big hole.
  structure  stage 1 at half size, 2x, then a two-step refine. Softer, and safer on very big holes.
The working picture is the input, optionally enlarged (Work Scale) because small crops failed at
their real size, then padded to the model's grid. Green #66FF00 goes under the grown mask as the
IC-LoRA reference, as LTX's workflow does.

Two changes from LTX's workflow, both from those tests:
  - Each blend goes over the model's own fill, not green. A narrow blend cannot overpower green in
    the low-frequency bands, so green came back.
  - The final result fades back to the input's exact float values, so beyond the fade values above
    1.0 survive and the picture stays a true match-move reference.
The model functions are passed in, so all of this can be tested without a model.
"""

from __future__ import annotations

import math
import os
import re
from typing import Callable

import numpy as np

GREEN = np.array([0x66, 0xFF, 0x00], dtype=np.float32) / 255.0
DETAIL, STRUCTURE = "detail", "structure"
GRID = {DETAIL: 32, STRUCTURE: 64}
MAX_LEVELS = 7
LOW_RES_LONG_SIDE = 64
MAX_LONG_SIDE = 6000  # as Alpha Gen: the LTX decoder refuses 6K and above
SMALL_SIDE = 512  # shorter sides than this did badly at real size in the tests
FADE_FRACTION = 0.06

Stage1 = Callable[[np.ndarray, int, str], np.ndarray]  # (H,W,3) 0-1 reference, seed, label -> same size
Stage2 = Callable[[np.ndarray, int], np.ndarray]  # (H,W,3) 0-1 picture, seed -> same size


# --------------------------------------------------------------------- pure helpers
def clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "plate"


def still_name(prefix: str, clip: str) -> str:
    """The start of a still's name: the optional prefix, then the clip's name on the timeline (or the
    comp's name in Fusion Studio), so a still can always be traced back to its shot."""
    joined = "_".join(part for part in (prefix, clip) if part)
    # Both can come from a comp someone else made, and they become a file name in the stills folder.
    return re.sub(r"[/\\:\x00-\x1f]", "_", joined).lstrip(".")


def variation_letter(variation: int) -> str:
    """0 is A, 25 is Z, 26 is AA."""
    letters = ""
    n = variation + 1
    while n > 0:
        n, rest = divmod(n - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


def variation_number(letters: str) -> int:
    n = 0
    for c in letters:
        n = n * 26 + ord(c) - ord("A") + 1
    return n - 1


def plate_file_name(name: str, frame: int, variation: int = 0) -> str:
    """The comp frame is the match-move reference, so it goes in the name and survives other tools.
    The name ends in a letter: Fusion takes names ending in a number as an image sequence, and its
    Loader then shows only one of the variations. Method and seed are in the file's metadata."""
    return f"{clean_name(name)}_LTXStill_f{frame:05d}_{variation_letter(variation)}.exr"


def next_variation(folders: list[str], name: str, frame: int) -> int:
    """The first unused letter for this frame, so a new Process never replaces earlier stills."""
    pattern = re.compile(re.escape(clean_name(name)) + rf"_LTXStill_f{frame:05d}_([A-Z]+)\.exr$")
    used = -1
    for folder in folders:
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for entry in names:
            found = pattern.fullmatch(entry)
            if found:
                used = max(used, variation_number(found.group(1)))
    return used + 1


def snap_size(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad_to_grid(a: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Edge-replicate (H,W) or (H,W,3) to the grid; returns the (top, left) offsets."""
    h, w = a.shape[:2]
    ph, pw = snap_size(h, multiple) - h, snap_size(w, multiple) - w
    top, left = ph // 2, pw // 2
    widths = [(top, ph - top), (left, pw - left)] + [(0, 0)] * (a.ndim - 2)
    return (np.pad(a, widths, mode="edge") if ph or pw else a), (top, left)


def green_composite(picture: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    return picture * (1.0 - m) + GREEN * m


def over_fill(picture: np.ndarray, mask: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """What the blend goes over: the picture, with the model's fill under the mask."""
    m = mask[..., None]
    return picture * (1.0 - m) + fill * m


def resize_lanczos(picture: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    if picture.shape[:2] == (height, width):
        return picture
    out = np.empty((height, width, 3), dtype=np.float32)
    for c in range(3):
        plane = Image.fromarray(picture[..., c].astype(np.float32), mode="F")
        out[..., c] = np.asarray(plane.resize((width, height), Image.Resampling.LANCZOS))
    return out


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area average going down; nearest going up, which keeps a mask's edge where it was drawn."""
    import torch
    import torch.nn.functional as F

    if mask.shape == (height, width):
        return mask
    t = torch.from_numpy(mask[None, None].astype(np.float32))
    return F.interpolate(t, size=(height, width), mode="area")[0, 0].numpy()


def grow_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if radius <= 0:
        return mask
    # Across, then down: the same result as one square pass, which took minutes on a 4K picture.
    size = radius * 2 + 1
    t = torch.from_numpy(mask[None, None].astype(np.float32))
    t = F.max_pool2d(t, kernel_size=(1, size), stride=1, padding=(0, radius))
    return F.max_pool2d(t, kernel_size=(size, 1), stride=1, padding=(radius, 0))[0, 0].numpy()


def _blur(x):  # noqa: ANN001, ANN202 - torch tensors (B,C,H,W)
    import torch
    import torch.nn.functional as F

    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=x.dtype) / 16.0
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)


def fade_width(width: int, height: int) -> int:
    """How far the fill fades out: 6 % of the long side (about 115 pixels at HD)."""
    return max(8, round(FADE_FRACTION * max(width, height)))


def fade_mask(mask: np.ndarray, solid: int, fade: int) -> np.ndarray:
    """1 up to `solid` pixels from the mask, then an eased fade to exactly 0 over `fade` pixels."""
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(mask < 0.5)
    t = np.clip((distance - solid) / max(1, fade), 0.0, 1.0).astype(np.float32)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _gaussian(x, levels: int) -> list:  # noqa: ANN001
    pyramid = [x]
    for _ in range(levels):
        pyramid.append(_blur(pyramid[-1])[..., ::2, ::2])
    return pyramid


def _up(x, size):  # noqa: ANN001, ANN202
    import torch.nn.functional as F

    return _blur(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


def _low_res_dilate(mask, radius: int):  # noqa: ANN001, ANN202
    """LTX's mask_low_res_dilation: dilate at long side 64, then back up."""
    import torch.nn.functional as F

    if radius <= 0:
        return mask
    h, w = mask.shape[-2:]
    scale = LOW_RES_LONG_SIDE / max(h, w)
    small = F.interpolate(mask, size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="bilinear",
                          align_corners=False, antialias=True)
    small = F.max_pool2d(small, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def laplacian_blend(fill: np.ndarray, base: np.ndarray, mask: np.ndarray, dilation: int) -> np.ndarray:
    """(H,W,3) in 0-1: `fill` inside the mask, `base` outside, joined per frequency band."""
    import torch
    import torch.nn.functional as F

    h, w = fill.shape[:2]
    pot_h, pot_w = 2 ** math.ceil(math.log2(h)), 2 ** math.ceil(math.log2(w))
    levels = min(MAX_LEVELS, int(math.log2(min(pot_h, pot_w))) - 2)  # the blur needs at least 4 pixels
    pad = (0, pot_w - w, 0, pot_h - h)
    m = _low_res_dilate(torch.from_numpy(mask[None, None].astype(np.float32)), dilation)
    a = torch.from_numpy(fill[None].astype(np.float32)).permute(0, 3, 1, 2)
    b = torch.from_numpy(base[None].astype(np.float32)).permute(0, 3, 1, 2)
    ga = _gaussian(F.pad(a, pad, mode="reflect"), levels)
    gb = _gaussian(F.pad(b, pad, mode="reflect"), levels)
    gm = _gaussian(F.pad(m, pad, mode="reflect"), levels)
    blended = ga[-1] * gm[-1] + gb[-1] * (1.0 - gm[-1])
    for i in range(levels - 1, -1, -1):
        size = ga[i].shape[-2:]
        la = ga[i] - _up(ga[i + 1], size)
        lb = gb[i] - _up(gb[i + 1], size)
        blended = _up(blended, size) + la * gm[i] + lb * (1.0 - gm[i])
    return blended[..., :h, :w].clamp(0.0, 1.0).permute(0, 2, 3, 1)[0].numpy()


# --------------------------------------------------------------------- the recipe
def check_job(picture: np.ndarray, mask: np.ndarray, method: str, work_scale: int) -> list[str]:
    """Refuse what cannot work; returns advice for what may work badly."""
    if method not in GRID:
        raise ValueError(f"unknown method {method}")
    if picture.shape[:2] != mask.shape:
        raise ValueError(f"picture is {picture.shape[1]}x{picture.shape[0]}, mask is {mask.shape[1]}x{mask.shape[0]}")
    if not np.nan_to_num(mask).max() > 0.5:
        raise ValueError("the mask is empty: white marks what to fill")
    h, w = picture.shape[:2]
    if max(w, h) * work_scale >= MAX_LONG_SIDE:
        raise ValueError(f"{w}x{h} at Work Scale {work_scale}x is too large for the LTX model (6K and above)")
    if min(w, h) * work_scale < SMALL_SIDE:
        return [f"{w}x{h} at Work Scale {work_scale}x is small for this model; a higher Work Scale may fill better"]
    return []


def make_plate(picture: np.ndarray, mask: np.ndarray, method: str, seed: int, stage1: Stage1, stage2: Stage2,
               work_scale: int = 1, grow: int = 16, blend: int = 1) -> np.ndarray:
    """(H,W,3) float picture and (H,W) mask -> (H,W,3) float plate. `grow` is in picture pixels."""
    check_job(picture, mask, method, work_scale)
    height, width = mask.shape
    clean_mask = np.nan_to_num(mask.astype(np.float32)).clip(0.0, 1.0)
    # The model only takes 0-1; anything outside that comes back from `picture` at the end.
    work = np.nan_to_num(picture.astype(np.float32)).clip(0.0, 1.0)
    work_w, work_h = width * work_scale, height * work_scale
    work = resize_lanczos(work, work_w, work_h).clip(0.0, 1.0)
    work_mask = resize_mask(clean_mask, work_w, work_h)
    padded, (top, left) = pad_to_grid(work, GRID[method])
    padded_mask, _ = pad_to_grid(work_mask, GRID[method])
    grown_px = grow * work_scale
    full_mask = grow_mask(padded_mask, grown_px)
    pad_h, pad_w = padded_mask.shape

    if method == DETAIL:
        filled = stage1(green_composite(padded, full_mask), seed, "detail")
    else:
        half_w, half_h = pad_w // 2, pad_h // 2
        half_mask = grow_mask(resize_mask(padded_mask, half_w, half_h), grown_px // 2)
        half = resize_lanczos(padded, half_w, half_h).clip(0.0, 1.0)
        rough = stage1(green_composite(half, half_mask), seed, "stage 1")
        joined = laplacian_blend(rough, over_fill(half, half_mask, rough), half_mask, blend)
        filled = stage2(resize_lanczos(joined, pad_w, pad_h).clip(0.0, 1.0), seed + 1)
    result = laplacian_blend(filled, over_fill(padded, full_mask, filled), full_mask, blend)
    result = result[top:top + work_h, left:left + work_w]
    if work_scale > 1:
        result = resize_lanczos(result, width, height)
    # The input's exact values beyond a long fade. The blend shifts brightness slightly a long way out,
    # and a short edge here showed as a line on the plain wall of the stairs shot. In the fade the
    # blend keeps the input's fine detail, so only that slow shift is being faded.
    soft = fade_mask(clean_mask, grow + 2, fade_width(width, height))[..., None]
    plate = picture.astype(np.float32) * (1.0 - soft) + result * soft
    outside = soft[..., 0] <= 0.0
    plate[outside] = picture[outside]
    return plate
