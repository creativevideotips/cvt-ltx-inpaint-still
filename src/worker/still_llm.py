# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""The Copy for LLM button: gets a frame and instructions ready for any LLM, to suggest a prompt.

  python -I -B still_llm.py <picture.bin> <mask.bin> <frame> <cache root> <input colour>

The area to fill goes flat grey, so the LLM can only describe what is around it, not what is being
removed. The instructions go on the clipboard and the picture is shown in Finder. Nothing is sent
anywhere: that is the user's choice, in whichever LLM they paste into.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ltxag_worker import oslayer, transport  # noqa: E402
from ltxag_worker.log import Logger  # noqa: E402
from ltxis_worker import hdr  # noqa: E402

log = Logger("llm")
TITLE = "LTX Inpaint Still"
LONG_SIDE = 1280  # plenty to describe a scene, and small enough for any LLM to accept
GREY = 0.5

INSTRUCTIONS = """\
The flat grey areas in the attached photograph are missing. An AI model will fill them in, and it needs a text prompt describing the whole scene.

Write one sentence of at most 35 words describing this place as it would look complete. Say what kind of place it is, then name its real surfaces, materials, colours and light, as you see them in the photograph.

Describe only what should be there. Never mention people, vehicles, lettering, the grey areas, or anything that was removed. Never use words like no, without, clean or remove, and don't describe a change.

Reply with only the sentence."""

DONE = ("The instructions are on the clipboard, and the picture is showing in Finder.\n\n"
        "In your LLM, paste the instructions and add the picture. Then put its sentence in the Prompt field, "
        "and check it's right for the shot.\n\n"
        "An online LLM receives this frame, so check that's allowed for this footage.")


def blanked(picture: np.ndarray, mask: np.ndarray, space: str = hdr.AS_IS) -> Image.Image:
    """The picture as a person would see it, with the area to fill grey, 8-bit, no longer than LONG_SIDE."""
    out = hdr.to_display(picture[..., :3], space).copy()
    out[mask > 0.5] = GREY
    image = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
    image.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
    return image


def main() -> int:
    picture_path, mask_path, frame, cache_root, space = sys.argv[1:6]
    try:
        picture = transport.read_frames(picture_path)[0].astype(np.float32)
        mask = transport.read_frames(mask_path)[0, ..., 0].astype(np.float32)
        folder = os.path.join(cache_root, "llm")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"frame_{int(frame):05d}_for_llm.png")
        blanked(picture, mask, space).save(path)
        subprocess.run(["/usr/bin/pbcopy"], input=INSTRUCTIONS.encode(), check=True, timeout=10)
        subprocess.run(["/usr/bin/open", "-R", path], check=True, timeout=10)
        log.info(f"copy for LLM: {path}")
        oslayer.message("info", TITLE, DONE, 420, True)
        return 0
    except Exception as exc:  # noqa: BLE001 - shown to the user
        log.error(f"copy for LLM failed: {exc}")
        oslayer.message("error", TITLE, f"Copy for LLM failed: {exc}", 420, True)
        return 1
    finally:
        for staged in (picture_path, mask_path):
            try:
                os.unlink(staged)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
