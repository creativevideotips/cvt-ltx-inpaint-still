# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""LTX Inpaint Still worker: makes clean plate stills. Runs in the LTX venv.

Built on Alpha Gen's worker (ltx_worker.py: Engine, Accel, Server), which loads the model, keeps
it resident and reports its steps. This file swaps the LoRA and runs one job at a time:

  one-shot:  still_worker.py --job JOB.json --model-dir M --embeddings E
  serve:     still_worker.py --serve SOCKET --model-dir M --embeddings E [--host-pid N]
             Each connection sends the job as one JSON line and receives the output lines back.

A job is {"job_id", "picture", "mask", "method", "seeds": [...], "work_scale", "grow", "blend",
"frame", "name", "out_dir", "precision", "model_dir", "unload"}. "picture" and "mask" are Alpha Gen's
staged frame files (ltxag_worker/transport.py) holding one frame: RGB and one channel. Each seed writes
<out_dir>/<plate name>.exr. Output lines: info <text>, step <label> <k> <n>, plate <seed> <path>
<seconds>, memory <peak allocated GB> <peak reserved GB>, done, error <kind> <text>.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import ltx_worker  # noqa: E402
from ltxag_worker import transport  # noqa: E402
from ltxag_worker.log import Logger  # noqa: E402
from ltxis_worker import exr, fill, hdr, person  # noqa: E402

log = Logger("still")

LORA_FILE = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
FRAME_RATE = ltx_worker.FRAME_RATE
STAGE_2_SIGMAS = [0.725, 0.421875, 0.0]  # LTX's inpaint workflow enters stage 2 here

_alpha_gen_paths = ltx_worker.model_paths_for


def model_paths_for(model_dir: str) -> dict:
    return {**_alpha_gen_paths(model_dir), "lora": f"{model_dir}/loras/{LORA_FILE}"}


ltx_worker.model_paths_for = model_paths_for  # the Engine looks it up here


def to_unit(video) -> np.ndarray:  # noqa: ANN001 - a torch tensor
    v = video.float()
    if not video.dtype.is_floating_point or float(v.max()) > 1.5:
        v = v / 255.0
    return v.clamp(0.0, 1.0).numpy()


class Model:
    """The two model passes make_plate needs, on a loaded Engine."""

    def __init__(self, engine, emit, float32_vae: bool = False) -> None:  # noqa: ANN001
        self.engine, self.emit, self.float32_vae = engine, emit, float32_vae

    def _vae_dtype(self):  # noqa: ANN202
        if self.float32_vae:  # LTX's HDR path: ACEScct values need the VAE's full precision
            return self.engine.torch.float32
        return self.engine.torch.bfloat16 if self.engine.fp16 else None

    def stage1(self, reference: np.ndarray, seed: int, label: str) -> np.ndarray:
        """The IC-LoRA pass, one frame, at the reference's own size (the pipeline's skip_stage_2)."""
        engine, torch = self.engine, self.engine.torch
        h, w = reference.shape[:2]
        # A copy: from_numpy shares memory, and mul_ would change the caller's picture.
        ltx_worker.MemoryFrameSource.frames = torch.from_numpy(reference[None].copy()).float().mul_(255.0)
        engine._on_step = lambda k, n: self.emit("step", label.replace(" ", "_"), k, n)
        try:
            with torch.inference_mode():
                decoded, _, _ = engine.pipeline.stream_chunks(
                    prompt="", seed=seed, height=h * 2, width=w * 2, num_frames=1, frame_rate=FRAME_RATE, images=[],
                    video_conditioning=[(ltx_worker.MEMORY_REF, 1.0)], skip_stage_2=True,
                    vae_dtype=self._vae_dtype())
                video = torch.cat([c.video.to("cpu") for c in decoded], dim=0)
        finally:
            ltx_worker.MemoryFrameSource.frames = None
        return to_unit(video)[0]

    def stage2(self, picture: np.ndarray, seed: int) -> np.ndarray:
        """Encode, then refine from sigma 0.725 with the LoRA on and no reference."""
        from dataclasses import replace

        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import AUTO_TILING
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.chunks.layout import ChunkConfig
        from ltx_pipelines.chunks.operators import decode_chunks, denoise_chunks
        from ltx_pipelines.chunks.planner import generate_uniform_chunks
        from ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae
        from ltx_pipelines.utils.media_io import video_preprocess
        from ltx_pipelines.utils.types import VideoAudio

        engine, torch = self.engine, self.engine.torch
        p = engine.pipeline
        h, w = picture.shape[:2]
        engine._on_step = lambda k, n: self.emit("step", "stage_2", k, n)
        host = torch.from_numpy(picture[None].copy()).float().mul_(255.0)
        target = VideoPixelShape(batch=1, frames=1, height=h, width=w, fps=FRAME_RATE)
        generator = torch.Generator(device=p.device).manual_seed(seed)
        with torch.inference_mode():
            video = video_preprocess((host[i : i + 1].to(p.device) for i in range(1)), h, w, p.dtype, p.device)
            latent = p.image_conditioner(lambda enc: enc(video))
            del video
            (context,) = p.prompt_encoder([""])
            chunks = generate_uniform_chunks(
                target=target, context=VideoAudio(video=context.video_encoding, audio=context.audio_encoding),
                device=p.device, dtype=p.dtype, config=ChunkConfig(chunk_pixel_frames=1, next_video_carry_frames=0),
                video_scale_factors=p._diffusion_stage.video_scale_factors)
            seeded = (replace(chunk, video=latent) for chunk in chunks)
            sigmas = torch.tensor(STAGE_2_SIGMAS, dtype=torch.float32, device=p.device)
            denoised = denoise_chunks(seeded, p._diffusion_stage.with_loras(p._loras), sigmas=sigmas,
                                      noiser=GaussianNoiser(generator=generator), fps=FRAME_RATE)
            tiling = ensure_tiling_config(
                AUTO_TILING, scale_factors=tiling_scale_factors_for_vae(p.video_decoder.checkpoint_path),
                video_shape=target, vae_checkpoint_path=p.video_decoder.checkpoint_path,
                diffvae_optimization=p.video_decoder.diffvae_optimization, device=p.device)
            decoded = decode_chunks(denoised, p.video_decoder, None, FRAME_RATE, tiling_config=tiling,
                                    generator=generator, dtype=self._vae_dtype() or p.dtype)
            out = torch.cat([c.video.to("cpu") for c in decoded], dim=0)
        return to_unit(out)[0]


def load_one_frame(path: str, channels: int) -> np.ndarray:
    frames = transport.read_frames(path)
    if frames.shape[0] != 1 or frames.shape[3] != channels:
        raise ValueError(f"{path}: expected 1 frame of {channels} channel(s), got {frames.shape}")
    return frames[0].astype(np.float32)


def encode_prompt(text: str, out: str, model_dir: str) -> int:
    """Child-process mode: run the text encoder once for a prompt, save, exit. Its own process, like
    ltx_worker.build_embeddings, so its 26 GB are really gone before the model needs them."""
    import torch
    from safetensors.torch import save_file

    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.model_paths import ModelPaths

    paths = model_paths_for(model_dir)
    model_paths = ModelPaths.from_split(transformer_path=paths["transformer"], text_encoder_path=paths["text_encoder"],
                                        video_vae_path=paths["video_vae"])
    encoder = PromptEncoder(model_paths, torch.bfloat16, ltx_worker.Accel(torch).device)
    with torch.inference_mode():
        (encoded,) = encoder([text])
    tensors = {"video_encoding": encoded.video_encoding, "attention_mask": encoded.attention_mask}
    if encoded.audio_encoding is not None:
        tensors["audio_encoding"] = encoded.audio_encoding
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp{os.getpid()}"
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, tmp, metadata={"prompt": text})
    os.replace(tmp, out)
    return 0


def load_encoding(path: str, engine):  # noqa: ANN001, ANN201
    from safetensors.torch import load_file

    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

    t = load_file(path, device=engine.accel.kind)
    return EmbeddingsProcessorOutput(t["video_encoding"], t.get("audio_encoding"), t["attention_mask"])


def make_stills(seeds: list[int], spare: int, make, find_people, save, emit) -> int:  # noqa: ANN001
    """Makes len(seeds) stills. One with a person in it is thrown away, and up to `spare` further seeds
    replace those. Returns the next seed not used. make(seed) -> plate, find_people(plate) -> (kinds,
    why it couldn't check), save(plate, seed, how many saved so far, seconds)."""
    want, saved = len(seeds), 0
    pending = list(seeds)
    next_seed = int(seeds[-1]) + 1
    checking = True
    while saved < want and pending:
        seed = int(pending.pop(0))
        t0 = time.time()
        plate = make(seed)
        if checking:
            kinds, problem = find_people(plate)
            if problem:
                emit("person_problem", problem)
                checking = False  # said once; the job carries on without it
            elif kinds:
                emit("rejected", seed, ",".join(sorted(set(kinds))), f"{time.time() - t0:.1f}")
                if spare > 0:
                    spare -= 1
                    pending.append(next_seed)
                    next_seed += 1
                continue
        save(plate, seed, saved, time.time() - t0)
        saved += 1
    return next_seed


def run_job(engine, job: dict, emit) -> None:  # noqa: ANN001
    picture = load_one_frame(job["picture"], 3)
    mask = load_one_frame(job["mask"], 1)[..., 0]
    method, work_scale = job.get("method", fill.DETAIL), int(job.get("work_scale", 1))
    grow, blend = int(job.get("grow", 16)), int(job.get("blend", 1))
    frame, name = int(job["frame"]), job.get("name", "plate")
    for advice in fill.check_job(picture, mask, method, work_scale):
        emit("info", advice)
    os.makedirs(job["out_dir"], exist_ok=True)
    space = job.get("input_colour", hdr.AS_IS)
    linear = space in hdr.LINEAR_SPACES
    model = Model(engine, emit, float32_vae=linear)
    engine.accel.reset_peaks()
    sampling = threading.Event()  # torch.mps has no peak statistics

    def sample() -> None:
        while not sampling.wait(0.25):
            engine.accel.sample()

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    # The pipeline asks for "", and gets whatever this holds: the job's prompt, then back to empty.
    encoder = engine.pipeline.prompt_encoder
    usual = encoder.output
    if job.get("prompt_embedding"):
        encoder.output = load_encoding(job["prompt_embedding"], engine)
        emit("info", "using the job's prompt")
    helper = job.get("person_helper", "")
    fill_area = fill.grow_mask(mask, grow)  # also the still's alpha: what was filled, with its overlap

    def make(seed: int) -> np.ndarray:
        if linear:
            return hdr.make_plate(picture, mask, space, method, seed, model.stage1, model.stage2,
                                  work_scale=work_scale, grow=grow, blend=blend)
        return fill.make_plate(picture, mask, method, seed, model.stage1, model.stage2,
                               work_scale=work_scale, grow=grow, blend=blend)

    def find_people(plate: np.ndarray) -> tuple[list[str], str]:
        return person.check(helper, hdr.to_display(plate, space), fill_area) if helper else ([], "")

    def save(plate: np.ndarray, seed: int, saved: int, seconds: float) -> None:
        letter_index = int(job.get("first_variation", 0)) + saved
        path = os.path.join(job["out_dir"], fill.plate_file_name(name, frame, letter_index))
        metadata = {"cvtPlateFrame": str(frame), "cvtPlateSeed": str(seed), "cvtPlateMethod": method,
                    "cvtPlateWorkScale": str(work_scale), "cvtPlateColour": space}
        exr.write_exr(path, plate, metadata, alpha=fill_area)
        emit("plate", seed, path, f"{seconds:.1f}")

    try:
        emit("next_seed", make_stills(job["seeds"], int(job.get("spare_seeds", 0)), make, find_people, save, emit))
    finally:
        encoder.output = usual
        sampling.set()
        sampler.join()
        engine._on_step = lambda k, n: None
    peak_allocated, peak_reserved = engine.accel.peaks_gb()
    emit("memory", f"{peak_allocated:.1f}", f"{peak_reserved:.1f}")


class Server(ltx_worker.Server):
    """Alpha Gen's resident server, with this effect's job in place of the matte job."""

    def handle(self, conn) -> str:  # noqa: ANN001
        conn.settimeout(None)
        f = conn.makefile("rwb", buffering=0)

        def emit(*parts) -> None:  # noqa: ANN002
            try:
                f.write((" ".join(str(p) for p in parts) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                raise ltx_worker.Cancelled() from exc

        try:
            job = json.loads(f.readline().decode() or "{}")
        except json.JSONDecodeError as exc:
            emit("error", "input", f"bad request: {exc}")
            return "ok"
        if job.get("ping"):
            emit("pong", self.a.precision, self.a.model_dir)
            return "ok"
        if job.get("precision") != self.a.precision or job.get("model_dir") != self.a.model_dir:
            emit("restart")
            return "restart"
        self.write_info("busy")
        log.job = job.get("job_id", "-")
        t0 = time.time()
        try:
            run_job(self.engine, job, emit)
            emit("done")
            log.info(f"job done in {time.time() - t0:.1f} s")
        except ltx_worker.Cancelled:
            log.warn("client disconnected mid-job; job abandoned")
        except Exception as exc:  # noqa: BLE001 - report everything to the client
            kind, msg = ltx_worker.classify(exc)
            log.error(f"job failed: {kind} {msg}\n{traceback.format_exc()}")
            try:
                emit("error", kind, msg)
            except ltx_worker.Cancelled:
                pass
            if kind == "oom":
                self.engine.drop_transformer()
        finally:
            log.job = "-"
            self.write_info("ready")
        return "unload" if job.get("unload", False) else "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", help="one-shot: a job file")
    ap.add_argument("--serve", help="resident: a Unix socket path")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--embeddings", default="")
    ap.add_argument("--encode-prompt", help="child process: encode this prompt to --out, then exit")
    ap.add_argument("--out", help="with --encode-prompt")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--host-pid", type=int, default=0, help="serve: exit when this process is gone")
    ap.add_argument("--idle-timeout", type=float, default=ltx_worker.IDLE_TIMEOUT_DEFAULT)
    a = ap.parse_args()
    if a.encode_prompt:
        return encode_prompt(a.encode_prompt, a.out, a.model_dir)
    if a.serve:
        a.flame_pid, a.rebuild_transformer, a.frames_on_gpu = a.host_pid, False, False
        return Server(a).serve()
    if not a.job:
        ap.error("give --job or --serve")

    def emit(*parts) -> None:  # noqa: ANN002
        print(" ".join(str(p) for p in parts), flush=True)

    t0 = time.time()
    try:
        with open(a.job) as f:
            job = json.load(f)
        engine = ltx_worker.Engine(a.model_dir, a.embeddings, a.precision, emit)
        emit("info", f"model ready after {time.time() - t0:.1f} s")
        run_job(engine, job, emit)
    except Exception as exc:  # noqa: BLE001 - report everything, with traceback
        kind, msg = ltx_worker.classify(exc)
        emit("error", kind, msg)
        traceback.print_exc(file=sys.stdout)
        return 2
    emit("info", f"job finished in {time.time() - t0:.1f} s")
    emit("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
