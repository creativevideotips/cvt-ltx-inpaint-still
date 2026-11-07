# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""Run a still job from the command line, standing in for the plugin. For development.

  python -B still_cli.py --source CLIP_OR_IMAGE --frame 53 --mask MASK.png --out-dir DIR \
      [--crop x,y,w,h] [--method detail|structure] [--seeds 1,2,3] [--work-scale 1] [--grow 16] [--blend 1]
      [--prompt TEXT] [--person-check]

It stages the picture and mask as the plugin will, in float, and starts the worker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "..", "worker")
sys.path.insert(0, WORKER)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ltxag_worker import transport  # noqa: E402
from ltxis_worker import person, prompt  # noqa: E402

MODEL_DIR = os.path.expanduser("~/LTX-2.5")
EMBEDDINGS = os.path.expanduser("~/.cache/ltx-alphagen/mattes/prompt-empty.safetensors")


def read_picture(path: str, frame: int) -> np.ndarray:
    if path.lower().endswith(".exr"):
        sys.path.insert(0, os.path.expanduser("~/.cache/ltx-inpaint-still/pylib"))
        import OpenEXR

        with OpenEXR.File(path, separate_channels=True) as f:
            channels = f.channels()
            return np.stack([channels[k].pixels for k in "RGB"], -1).astype(np.float32)
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    import av

    with av.open(path) as container:
        for i, f in enumerate(container.decode(container.streams.video[0])):
            if i == frame:
                # 16-bit so a 10-bit source keeps its values, as float reaches the plugin.
                return f.to_ndarray(format="rgb48le").astype(np.float32) / 65535.0
    raise ValueError(f"no frame {frame} in {path}")


def stage(path: str, frames: np.ndarray) -> None:
    n, h, w, c = frames.shape
    with open(path, "wb") as f:
        f.write(transport.HEADER.pack(transport.IN_MAGIC, transport.VERSION, w, h, n, c))
        np.ascontiguousarray(frames, dtype="<f2").tofile(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crop", help="x,y,w,h, like a Crop before the node")
    ap.add_argument("--method", default="detail", choices=["detail", "structure"])
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--work-scale", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--grow", type=int, default=16)
    ap.add_argument("--blend", type=int, default=1)
    ap.add_argument("--name", default="plate")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--person-check", action="store_true")
    ap.add_argument("--colour", default="as_is", choices=["as_is", "acescg", "srgb_linear"])
    a = ap.parse_args()
    picture = read_picture(a.source, a.frame)
    with Image.open(a.mask) as im:
        mask = np.asarray(im.convert("RGB"), dtype=np.float32)[..., 0] / 255.0
    if a.crop:
        x, y, w, h = (int(v) for v in a.crop.split(","))
        picture, mask = picture[y:y + h, x:x + w], mask[y:y + h, x:x + w]
    with tempfile.TemporaryDirectory() as tmp:
        picture_path, mask_path = os.path.join(tmp, "picture.bin"), os.path.join(tmp, "mask.bin")
        stage(picture_path, picture[None])
        stage(mask_path, mask[None, ..., None])
        job = {"job_id": "cli", "picture": picture_path, "mask": mask_path, "method": a.method,
               "seeds": [int(s) for s in a.seeds.split(",")], "work_scale": a.work_scale, "grow": a.grow,
               "blend": a.blend, "frame": a.frame, "name": a.name, "out_dir": a.out_dir, "input_colour": a.colour}
        cache = os.path.expanduser("~/.cache/ltx-inpaint-still")
        text = prompt.clean(a.prompt)
        if text:
            job["prompt_embedding"] = prompt.embedding_path(cache, text, MODEL_DIR)
            if not os.path.exists(job["prompt_embedding"]):
                subprocess.run([sys.executable, "-I", "-B", os.path.join(WORKER, "still_worker.py"), "--model-dir", MODEL_DIR,
                                "--encode-prompt", text, "--out", job["prompt_embedding"]], check=True)
        if a.person_check:
            job["person_helper"], why = person.helper_path(cache)
            job["spare_seeds"] = max(3, len(job["seeds"]))
            print("person check helper:", job["person_helper"] or why, flush=True)
        job_path = os.path.join(tmp, "job.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        return subprocess.run([sys.executable, "-B", os.path.join(WORKER, "still_worker.py"), "--job", job_path,
                               "--model-dir", MODEL_DIR, "--embeddings", EMBEDDINGS]).returncode


if __name__ == "__main__":
    sys.exit(main())
