# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""The Open Stills Folder button: works out where stills will go, makes that folder and shows it.

  python -I -B still_open_folder.py <host> <node id> <stills folder>

It runs detached, because it asks the host where a path map points, and the host can't answer while
the plugin waits for it inside a button press. So you can check the location before running a job.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ltxag_worker import oslayer  # noqa: E402
from ltxag_worker.log import Logger  # noqa: E402
from ltxis_worker import fusion_link  # noqa: E402

log = Logger("folder")
TITLE = "LTX Inpaint Still"


def prepare(host: str, node_id: str, folder: str, fuscript: str | None = None) -> tuple[str, str]:
    """Makes the folder stills will go in. Returns it, and what to tell the user if it isn't the one asked for."""
    where = fusion_link.locate(host, node_id, folder, fuscript)
    os.makedirs(where["folder"], exist_ok=True)
    note = "" if where["wanted"] else f"Stills will go here because {where['problem']}."
    return where["folder"], note


def main() -> int:
    host, node_id, folder = sys.argv[1:4]
    log.node = node_id
    try:
        target, note = prepare(host, node_id, folder)
    except OSError as exc:
        log.error(f"could not make the folder: {exc}")
        oslayer.message("error", TITLE, f"Could not make the stills folder: {exc}", 420, True)
        return 1
    log.info(f"opening {target}; {note or 'the folder asked for'}")
    subprocess.run(["/usr/bin/open", target], check=False)
    if note:
        oslayer.message("info", TITLE, note, 420, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
