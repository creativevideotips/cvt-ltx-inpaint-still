# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""Where a prompt's stored encoding lives. Encoding runs the text encoder (about 26 GB, in its own
process, see still_worker.py), so each prompt is done once and kept. Standard library only."""

from __future__ import annotations

import hashlib
import os

TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
BIG_MEMORY_GB = 96  # below this, the idle model is unloaded first so the encoder never runs beside it


def clean(text: str) -> str:
    """One line, single spaces: a stray line break or space shouldn't mean a new encoding."""
    return " ".join((text or "").split())


def embedding_path(cache_root: str, text: str, model_dir: str) -> str:
    """Tied to the text encoder's file too, so a different encoder never reuses an old result."""
    st = os.stat(os.path.join(model_dir, TEXT_ENCODER))
    key = f"{st.st_size}|{text}".encode()
    return os.path.join(cache_root, "prompts", f"prompt-{hashlib.sha1(key).hexdigest()[:16]}.safetensors")


def needs_room_first() -> bool:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") < BIG_MEMORY_GB * 1024 ** 3
