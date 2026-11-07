# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""Scene-linear stills: ACEScg or linear sRGB in, the same space out.

LTX's native HDR path: the model works on ACEScct values, with the VAE at float32. ACEScct takes
ACEScg from 0 up to about 222. As in LTX's HDR inpaint workflow, the recipe in fill.py runs on those
values unchanged: the green stays its display value, because that is what the LoRA was trained on,
and the blend works on the ACEScct values as they are. Constants are AMPAS S-2016-001; the matrices
are the ones ltx_core uses (colour-science, Bradford).
"""

from __future__ import annotations

import numpy as np

from . import fill

AS_IS = "as_is"
ACESCG = "acescg"
SRGB_LINEAR = "srgb_linear"
LINEAR_SPACES = (ACESCG, SRGB_LINEAR)

_A_LIN = 10.5402377416545
_B_LIN = 0.0729055341958355
_X_BRK = 0.0078125
_Y_BRK = 0.155251141552511
_LOG_M = 17.52
_LOG_B = 9.72
_ROLLOFF = 0.6  # the fallback preview only: below this nothing changes
OCIO_CONFIG = "studio-config-latest"  # built into OCIO, so no file to find or ship
OCIO_DISPLAY = "sRGB - Display"
OCIO_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
OCIO_SPACES = {ACESCG: "ACEScg", SRGB_LINEAR: "Linear Rec.709 (sRGB)"}
_processors: dict = {}

ACESCG_TO_SRGB = np.array([[1.70505099265798, -0.621792120657, -0.08325887200098],
                           [-0.13025641750704, 1.1408047365754, -0.01054831906836],
                           [-0.02400335680462, -0.12896897606497, 1.15297233286959]], np.float32)
SRGB_TO_ACESCG = np.array([[0.61309740240119, 0.33952314618411, 0.04737945141471],
                           [0.07019372246958, 0.91635387905734, 0.01345239847307],
                           [0.02061559288223, 0.10956977293814, 0.86981463417964]], np.float32)


def _matrix(picture: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return picture @ matrix.T


def to_acescct(linear: np.ndarray, space: str) -> np.ndarray:
    """Scene-linear in `space` -> ACEScct, 0 to 1. Negative values go to 0, as in LTX's own loader."""
    acescg = _matrix(linear, SRGB_TO_ACESCG) if space == SRGB_LINEAR else linear
    x = np.clip(np.nan_to_num(acescg.astype(np.float32)), 0.0, None)
    log_part = (np.log2(np.maximum(x, 1e-12)) + _LOG_B) / _LOG_M
    return np.clip(np.where(x > _X_BRK, log_part, _A_LIN * x + _B_LIN), 0.0, 1.0).astype(np.float32)


def from_acescct(acescct: np.ndarray, space: str) -> np.ndarray:
    """ACEScct -> scene-linear in `space`."""
    ct = np.clip(acescct.astype(np.float32), 0.0, 1.0)
    acescg = np.where(ct > _Y_BRK, np.power(2.0, ct * _LOG_M - _LOG_B), (ct - _B_LIN) / _A_LIN).astype(np.float32)
    return _matrix(acescg, ACESCG_TO_SRGB) if space == SRGB_LINEAR else acescg


def to_display(picture: np.ndarray, space: str) -> np.ndarray:
    """What a person would see, for the check for people and Copy for LLM: OCIO's ACES 2.0 SDR view on
    an sRGB display, so highlights above 1.0 still show. Only for looking at: it never reaches a still."""
    if space not in LINEAR_SPACES:
        return np.clip(picture, 0.0, 1.0)
    processor = _ocio_processor(space)
    if processor is not None:
        out = np.array(picture[..., :3], np.float32)
        processor.applyRGB(out)
        return np.clip(out, 0.0, 1.0)
    return _rolled_off(picture, space)


def _ocio_processor(space: str):  # noqa: ANN202
    """The CPU processor for a space, kept for next time. None if OCIO can't give one."""
    if space not in _processors:
        try:
            import PyOpenColorIO as ocio  # in the LTX runtime

            config = ocio.Config.CreateFromBuiltinConfig(OCIO_CONFIG)
            view = ocio.DisplayViewTransform()
            view.setSrc(OCIO_SPACES[space])
            view.setDisplay(OCIO_DISPLAY)
            view.setView(OCIO_VIEW)
            _processors[space] = config.getProcessor(view).getDefaultCPUProcessor()
        except Exception:  # noqa: BLE001 - a preview must never stop a job; the fallback still shows highlights
            _processors[space] = None
    return _processors[space]


def _rolled_off(picture: np.ndarray, space: str) -> np.ndarray:
    """If OCIO isn't there: midtones as they are, highlights rolled off."""
    rec709 = _matrix(picture, ACESCG_TO_SRGB) if space == ACESCG else picture
    x = np.clip(rec709, 0.0, None)
    t = (x - _ROLLOFF) / (1.0 - _ROLLOFF)
    x = np.where(x <= _ROLLOFF, x, _ROLLOFF + (1.0 - _ROLLOFF) * t / (1.0 + t))
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def make_plate(picture: np.ndarray, mask: np.ndarray, space: str, method: str, seed: int, stage1, stage2,  # noqa: ANN001
               work_scale: int = 1, grow: int = 16, blend: int = 1) -> np.ndarray:
    """(H,W,3) scene-linear picture -> plate in the same space. Beyond the fade, the input's exact values:
    those can be negative or above ACEScct's range, and only here do they come back."""
    plate = fill.make_plate(to_acescct(picture, space), mask, method, seed, stage1, stage2, work_scale=work_scale,
                            grow=grow, blend=blend)
    decoded = from_acescct(plate, space)
    height, width = mask.shape
    clean = np.nan_to_num(mask.astype(np.float32)).clip(0.0, 1.0)
    soft = fill.fade_mask(clean, grow + 2, fill.fade_width(width, height))[..., None]
    out = picture.astype(np.float32) * (1.0 - soft) + decoded * soft
    outside = soft[..., 0] <= 0.0
    out[outside] = picture[outside]
    return out
