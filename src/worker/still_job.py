# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""One Process press: runs outside the host, launched detached by the plugin.

  python -I -B still_job.py <manifest.json> [--no-ui] [--stand-in]

It shows the progress window, asks the resident worker for the plates (starting it if needed), and
keeps <cache_root>/jobs/<job_id>.json current, which the plugin follows. The status file holds state
(running, done, failed, cancelled), message, and when done paths: the plates, comma-separated.
Then it cleans up its staged files. The worker connection follows Alpha Gen's worker/job.py.

--stand-in makes plates from a flat grey fill, with no model, so the plumbing can be tested.
"""

from __future__ import annotations

import fcntl
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ltxag_worker import oslayer  # noqa: E402
from ltxag_worker.log import Logger  # noqa: E402
from ltxis_worker import exr, fill, fusion_link, person, prompt  # noqa: E402

log = Logger("job")
EOF_LINE = None


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Status:
    def __init__(self, root: str, job_id: str) -> None:
        self.path = os.path.join(root, "jobs", f"{job_id}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, state: str, message: str = "", paths: list[str] | None = None, next_seed: int | None = None) -> None:
        tmp = f"{self.path}.tmp{os.getpid()}"
        fields = {"state": state, "message": message.replace('"', "'"), "paths": ",".join(paths or [])}
        if next_seed is not None:
            fields["next_seed"] = str(next_seed)
        with open(tmp, "w") as f:
            json.dump(fields, f)
        os.replace(tmp, self.path)


class Progress:
    """The window's line and bar: which plate, which step, time so far and time left."""

    STEPS = {"detail": 8, "structure": 10}

    def __init__(self, m: dict, ui: bool) -> None:
        self.win = oslayer.ProgressWindow(m.get("title", "LTX Inpaint Still"), 560, ui)
        self.total = len(m["seeds"])
        self.steps = self.STEPS.get(m.get("method"), 8)
        self.done = 0
        self.step = 0
        self.t0 = time.time()
        self.plate_times: list[float] = []
        self.note = "Starting the worker"
        self.lock = threading.Lock()
        self.finished = False

    def text(self) -> tuple[float, str]:
        with self.lock:
            elapsed = time.time() - self.t0
            if self.done == 0 and self.step == 0:
                return 0.0, f"{self.note} · {mmss(elapsed)} elapsed"
            plate = min(self.done + 1, self.total)
            frac = (self.done + min(self.step / self.steps, 0.95)) / self.total
            line = f"Still {plate} of {self.total} · step {self.step}/{self.steps} · {mmss(elapsed)} elapsed"
            if self.plate_times:
                each = sum(self.plate_times) / len(self.plate_times)
                left = max(0.0, each * (self.total - self.done) - (time.time() - self.plate_t0))
                line += f" · about {mmss(left)} left"
            return frac, line

    plate_t0 = 0.0

    def tick(self) -> None:
        frac, line = self.text()
        with self.lock:
            if self.finished or not self.win.shown or self.win.gone():
                return
            try:
                self.win.show(frac, line)
            except BrokenPipeError:
                pass

    def cancelled(self) -> bool:
        return not self.finished and self.win.gone()

    def close(self) -> None:
        with self.lock:
            self.finished = True
        self.win.close(wait=5)


class ResidentWorker:
    """Starts the resident worker when needed, then streams one job from it."""

    def __init__(self, m: dict, job: dict, progress: Progress) -> None:
        self.sock_path = m["worker_socket"]
        self.info_path = self.sock_path + ".json"
        self.m, self.progress = m, progress
        self.lines: queue.Queue = queue.Queue()
        self.server_pid = 0
        self.proc = None
        for _ in range(3):
            self._ensure_server()
            conn = self._connect()
            f = conn.makefile("rb")
            try:
                conn.sendall((json.dumps(job) + "\n").encode())
                first = f.readline().decode().rstrip("\n")
            except (BrokenPipeError, ConnectionResetError):
                first = ""
            if first in ("restart", ""):
                log.info("the resident worker has other settings or exited; starting another")
                conn.close()
                if self.proc is not None:
                    self.proc.wait(timeout=60)
                    self.proc = None
                else:
                    deadline = time.time() + 60
                    while pid_alive(self.server_pid) and time.time() < deadline:
                        time.sleep(0.2)
                continue
            self.lines.put(first)

            def pump() -> None:
                for raw in f:
                    self.lines.put(raw.decode().rstrip("\n"))
                self.lines.put(EOF_LINE)

            self.conn = conn
            threading.Thread(target=pump, daemon=True).start()
            return
        raise RuntimeError("could not get a resident worker with matching settings")

    def _info(self) -> dict:
        try:
            with open(self.info_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _ensure_server(self) -> None:
        os.makedirs(os.path.dirname(self.sock_path), exist_ok=True)
        with open(self.sock_path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            info = self._info()
            if info.get("pid") and pid_alive(info["pid"]) and info.get("state") in ("loading", "ready", "busy"):
                self.server_pid = info["pid"]
                log.info(f"reusing resident worker pid {self.server_pid}")
                return
            m = self.m
            cmd = [m["worker_python"], "-I", "-B", m["worker"], "--serve", self.sock_path, "--precision", m["precision"],
                   "--host-pid", str(m.get("host_pid", 0)), "--model-dir", m["model_dir"], "--embeddings", m["embeddings"]]
            out = open(os.path.join(os.path.dirname(self.sock_path), "worker.out"), "w")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=True)  # outlives this job, to stay loaded
            self.server_pid = self.proc.pid
            log.info(f"started resident worker pid {self.server_pid}")
            deadline = time.time() + 120
            while time.time() < deadline and self._info().get("pid") != self.server_pid:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"the worker exited at start (exit {self.proc.returncode}); see run/worker.out")
                time.sleep(0.2)

    def _alive(self) -> bool:
        if self.proc is not None and self.proc.pid == self.server_pid:
            return self.proc.poll() is None  # a killed child stays a zombie until reaped
        return pid_alive(self.server_pid)

    def _connect(self) -> socket.socket:
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.progress.cancelled():
                raise KeyboardInterrupt
            info = self._info()
            if info.get("state") == "failed":
                raise RuntimeError(f"the worker failed to load: {info.get('message')}")
            if self.server_pid and not self._alive():
                raise RuntimeError(f"the worker died while loading; see run/worker.out")
            if info.get("state") == "loading":
                self.progress.note = "Loading the model"
            elif info.get("state") == "busy":
                self.progress.note = "Waiting for another job"
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.sock_path)
                return s
            except (FileNotFoundError, ConnectionRefusedError):
                s.close()
                time.sleep(0.3)
        raise RuntimeError("timed out waiting for the worker")

    def kill(self) -> None:
        """Cancel must really free the GPU, so the worker goes, and the next Process reloads it."""
        try:
            os.killpg(self.server_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.time() + 10
        while self._alive() and time.time() < deadline:
            time.sleep(0.1)


class StandIn:
    """No model: runs the real recipe with a flat grey fill, for testing the plumbing."""

    def __init__(self, m: dict, job: dict) -> None:
        self.lines: queue.Queue = queue.Queue()

        def run() -> None:
            import numpy as np

            from ltxag_worker import transport

            try:
                picture = transport.read_frames(job["picture"])[0].astype(np.float32)
                mask = transport.read_frames(job["mask"])[0, ..., 0].astype(np.float32)
                os.makedirs(job["out_dir"], exist_ok=True)
                for variation, seed in enumerate(job["seeds"]):
                    plate = fill.make_plate(picture, mask, job["method"], seed,
                                            lambda r, _s, _l: np.full_like(r, 0.5), lambda p, _s: p,
                                            work_scale=job["work_scale"], grow=job["grow"], blend=job["blend"])
                    path = os.path.join(job["out_dir"], fill.plate_file_name(job["name"], job["frame"],
                                                                             job.get("first_variation", 0) + variation))
                    exr.write_exr(path, plate, alpha=fill.grow_mask(mask, job["grow"]))
                    self.lines.put(f"plate {seed} {path} 0.1")
                self.lines.put("done")
            except Exception as exc:  # noqa: BLE001 - becomes the job's failure
                self.lines.put(f"error internal {exc}")
            self.lines.put(EOF_LINE)

        threading.Thread(target=run, daemon=True).start()

    def kill(self) -> None:
        pass


def stop_idle_worker(socket_path: str) -> None:
    """Makes room on a smaller Mac. A worker that is busy with another job is left alone."""
    try:
        with open(socket_path + ".json") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if info.get("state") == "ready" and info.get("pid") and pid_alive(info["pid"]):
        log.info(f"unloading the idle model before encoding the prompt (pid {info['pid']})")
        os.kill(info["pid"], signal.SIGTERM)
        deadline = time.time() + 30
        while pid_alive(info["pid"]) and time.time() < deadline:
            time.sleep(0.2)


def encode_prompt(m: dict, text: str, path: str, progress: Progress) -> None:
    """Runs the text encoder in its own process. Raises if it fails; KeyboardInterrupt on Cancel."""
    if prompt.needs_room_first():
        stop_idle_worker(m["worker_socket"])
    progress.note = "Reading the prompt (each new prompt once)"
    t0 = time.time()
    logs = os.path.join(os.path.dirname(m["worker_socket"]), "prompt.out")
    os.makedirs(os.path.dirname(logs), exist_ok=True)
    with open(logs, "w") as out:
        proc = subprocess.Popen([m["worker_python"], "-I", "-B", m["worker"], "--model-dir", m["model_dir"],
                                 "--encode-prompt", text, "--out", path], stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if progress.cancelled():
                proc.kill()
                proc.wait()
                raise KeyboardInterrupt
            time.sleep(0.2)
    if proc.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"could not read the prompt (exit {proc.returncode}); see run/prompt.out")
    log.info(f"prompt encoded in {time.time() - t0:.1f} s: {path}")


def dwab_converter(m: dict) -> tuple:
    """A function that writes the smaller saved copy, or None and the reason. Never raises."""
    library = os.path.join(m["cache_root"], "pylib")
    uv = os.path.join(m.get("model_dir", ""), "runtime", "bin", "uv")
    try:
        openexr, why = exr.load_openexr(library, uv)
    except Exception as exc:  # noqa: BLE001 - the stills are made; this only decides their saved format
        return None, str(exc)[:120]
    if openexr is None:
        return None, why
    level = float(m.get("dwab_level", 45))

    def convert(source: str, target: str) -> None:
        picture, alpha, metadata = exr.read_exr_with_alpha(source)
        exr.write_dwab(openexr, target, picture, level, metadata, alpha=alpha)

    return convert, ""


def main() -> int:
    manifest_path = sys.argv[1]
    ui, stand_in = "--no-ui" not in sys.argv, "--stand-in" in sys.argv
    with open(manifest_path) as f:
        m = json.load(f)
    log.node, log.job = m.get("node_id", "-"), m["job_id"]
    status = Status(m["cache_root"], m["job_id"])
    status.write("running", f"Processing frame {m['frame']}")
    progress = Progress(m, ui)
    t0 = time.time()
    job = {k: m[k] for k in ("job_id", "picture", "mask", "method", "seeds", "work_scale", "grow", "blend", "frame",
                             "name", "out_dir", "precision", "model_dir")}
    job["input_colour"] = m.get("input_colour", "as_is")
    # Where plates go and the comp's name, asked of the host before the slow part.
    where = None
    if m.get("plate_folder") and not stand_in:
        where = fusion_link.locate(m.get("host", "fusion"), m.get("node_id", ""), m["plate_folder"])
        log.info(f"plates go to {where['folder']} (as asked: {where['wanted']}); comp '{where['comp_name']}'; "
                 f"problem: {where['problem'] or 'none'}")
        job["name"] = fill.still_name(job["name"], where["comp_name"])
    # Letters carry on from what is already there, so earlier stills are never replaced.
    folders = [job["out_dir"]] + ([where["folder"]] if where else [])
    job["first_variation"] = fill.next_variation(folders, job["name"], job["frame"])
    log.info(f"start: frame {m['frame']} {m['method']} seeds {m['seeds']} scale {m['work_scale']} grow {m['grow']}")
    ticking = threading.Event()

    def tick() -> None:
        while not ticking.wait(1.0):
            progress.tick()

    threading.Thread(target=tick, daemon=True).start()
    state, message, paths = "failed", "the worker ended without a result", []
    worker = None
    next_seed = None
    rejected: list[str] = []
    person_note = ""
    try:
        text = prompt.clean(m.get("prompt", ""))
        if text and not stand_in:
            job["prompt_embedding"] = prompt.embedding_path(m.get("work_root") or m["cache_root"], text, m["model_dir"])
            log.info(f"prompt: {len(text)} characters")  # support logs get sent to people; the words stay out
            if not os.path.exists(job["prompt_embedding"]):
                encode_prompt(m, text, job["prompt_embedding"], progress)
        if m.get("person_check") and not stand_in:
            progress.note = "Getting the person check ready"
            job["person_helper"], why = person.helper_path(m["cache_root"])
            job["spare_seeds"] = max(3, len(m["seeds"]))  # about one minute each; a limit, so a job always ends
            if why:
                person_note = f"Stills were not checked for people: {why}."
                log.warn(person_note)
        worker = StandIn(m, job) if stand_in else ResidentWorker(m, job, progress)
        progress.plate_t0 = time.time()
    except KeyboardInterrupt:
        state, message = "cancelled", "cancelled"
    except Exception as exc:  # noqa: BLE001 - becomes the job's failure
        log.error(f"could not start the worker: {exc}")
        message = str(exc)
    while worker is not None:
        if progress.cancelled():
            log.info("cancel pressed: stopping the worker")
            worker.kill()
            state, message = "cancelled", "cancelled"
            break
        try:
            line = worker.lines.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is EOF_LINE:
            break
        if line.startswith("step "):
            with progress.lock:
                label, k = line.split()[1:3]
                progress.step = int(k) + (8 if label == "stage_2" else 0)
        elif line.startswith("plate "):
            _, seed, rest = line.split(" ", 2)
            path, seconds = rest.rsplit(" ", 1)
            paths.append(path)
            with progress.lock:
                progress.done += 1
                progress.step = 0
                progress.plate_times.append(float(seconds))
                progress.plate_t0 = time.time()
            status.write("running", f"Made {len(paths)} of {len(m['seeds'])} stills")
            log.info(f"plate seed {seed}: {path} in {seconds} s")
        elif line.startswith("rejected "):
            _, seed, kinds, seconds = line.split(" ", 3)
            rejected.append(seed)
            progress.note = f"Discarded seed {seed}, which had a person; trying another"
            log.info(f"discarded seed {seed}: found {kinds}, after {seconds} s")
            with progress.lock:
                progress.step = 0
                progress.plate_times.append(float(seconds))
                progress.plate_t0 = time.time()
            status.write("running", f"Discarded seed {seed}, which had a person; trying another")
        elif line.startswith("person_problem "):
            person_note = f"The check for people stopped: {line.split(' ', 1)[1]}."
            log.warn(person_note)
        elif line.startswith("next_seed "):
            next_seed = int(line.split()[1])
        elif line == "done":
            state, message = "done", ""
        elif line.startswith("error "):
            state, message = "failed", line[6:]
            log.error(f"worker: {line}")
        elif line.split(" ", 1)[0] in ("info", "memory"):
            log.info(f"worker: {line}")
        else:
            log.warn(f"worker output: {line}")
    ticking.set()
    progress.close()
    if state == "done" and rejected and not paths:
        state = "failed"
        message = (f"Every still had a person and was discarded (seeds {', '.join(rejected)}), so none were saved. "
                   "Try a prompt that describes the scene, a smaller mask, or Structure.")
    if state == "done":
        count = len(paths)
        message = f"Made {count} still{'s' if count != 1 else ''} from frame {m['frame']} in {mmss(time.time() - t0)}."
        if rejected:
            message += f" Discarded {len(rejected)} with people (seed{'s' if len(rejected) != 1 else ''} {', '.join(rejected)})."
        if count < len(m["seeds"]):
            message += f" Only {count} of {len(m['seeds'])}: try a prompt that describes the scene, or a smaller mask."
        if person_note:
            message += " " + person_note
        if m.get("cache_note"):
            message += " " + m["cache_note"]
        if where:  # never fails the job: the stills are made, and the node can show them
            status.write("running", "Saving stills")
            convert, why = dwab_converter(m)
            paths, note = fusion_link.deliver(where, m.get("host", "fusion"), m.get("node_id", ""), paths, convert, why)
            log.info(f"delivered: {paths}; {note or 'all as asked'}")
            if note:
                message += " " + note
    status.write(state, message, paths, next_seed)
    for p in (m["picture"], m["mask"], manifest_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    log.info(f"end: {state} {message}")
    if state == "failed":
        oslayer.message("error", "LTX Inpaint Still", f"Processing failed: {message}", 420, ui)
    return 0 if state == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
