# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""Saves plates beside the comp and reloads them with Loaders, through Fusion's scripting.

An OFX plugin can't see its comp's path or add tools, but Fusion's scripting can, from outside the
host, with the fuscript that ships with it. The node is found by the id saved in its nodeId input.
If this can't be done (scripting off in Resolve, a comp never saved) the plates still go to a fallback
folder, and the node's output never depends on any of this. Standard library only.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess

FALLBACK_FOLDER = "~/Documents/LTX Inpaint Stills"
PLUGIN_ID = "com.creativevideotips.ltx.inpaintstill"
TIMEOUT_SECONDS = 30
PREFIX = "LTXIS "

RESOLVE_FUSCRIPT = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fuscript"
FUSION_FUSCRIPTS = "/Applications/Blackmagic Fusion */Fusion.app/Contents/Libraries/fuscript"


def find_fuscript(host: str) -> str:
    if host == "resolve":
        return RESOLVE_FUSCRIPT if os.path.exists(RESOLVE_FUSCRIPT) else ""
    found = sorted(glob.glob(FUSION_FUSCRIPTS))
    return found[-1] if found else ""


def lua_string(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
    return f'"{out}"'


def _lua_header(host: str, node_id: str) -> str:
    connect = ('local app = bmd.scriptapp("Resolve", "localhost")\nlocal fu = app and app:Fusion()\n'
               if host == "resolve" else 'local fu = bmd.scriptapp("Fusion", "localhost")\n')
    return connect + f"""
local function say(s) io.write("{PREFIX}" .. s .. "\\n") end
if not fu then say("NOHOST") os.exit(0) end
local nodeId, pluginId = {lua_string(node_id)}, {lua_string(PLUGIN_ID)}
local function search(c)
  for _, t in pairs(c:GetToolList(false)) do
    if string.find(t:GetAttrs().TOOLS_RegID, pluginId, 1, true) and t:GetInput("nodeId") == nodeId then return t end
  end
end
local comp, node
local comps = fu:GetCompList() or {{}}
for _, c in pairs(comps) do
  node = search(c)
  if node then comp = c break end
end
if not comp then
  local c = fu:GetCurrentComp()
  node = c and search(c)
  if node then comp = c end
end
if not comp then say("NONODE") os.exit(0) end
"""


def locate_lua(host: str, node_id: str, folder: str) -> str:
    # On Resolve's Fusion page a comp has no file, so the clip under the playhead names the stills.
    clip = """
local ok, name = pcall(function()
  return app:GetProjectManager():GetCurrentProject():GetCurrentTimeline():GetCurrentVideoItem():GetName()
end)
if ok and name then say("CLIP " .. name) end
""" if host == "resolve" else ""
    return _lua_header(host, node_id) + f"""
say("MAPPED " .. comp:MapPath({lua_string(folder)}))
say("COMP " .. (comp:GetAttrs().COMPS_FileName or ""))
""" + clip


# The first row below the node with room for the whole batch, so a later Process never stacks its
# Loaders on the last ones. Tools can sit anywhere once they have been dragged, so this measures
# overlap in the flow's units instead of comparing grid cells. A tool is about 1 wide, and with its
# thumbnail showing about 1.6 tall, and more when a long name wraps, so rows are 3 apart.
ROW_FOR_LUA = """
local function rowFor(x, y, count, taken)
  local row = y + 3
  while true do
    local free = true
    for i = 1, count do
      for _, t in ipairs(taken) do
        if math.abs(t[1] - (x + 1 + i)) < 1.05 and math.abs(t[2] - row) < 3 then free = false end
      end
    end
    if free then return row end
    row = row + 3
  end
end
"""


def loaders_lua(host: str, node_id: str, plates: list[tuple[str, str]], dry_run: bool = False) -> str:
    """dry_run only reports where each Loader would go, and changes nothing in the comp."""
    entries = ",\n".join(f"{{{lua_string(path)}, {lua_string(name)}}}" for path, name in plates)
    return _lua_header(host, node_id) + f"""
local plates = {{{entries}}}
local attrs = comp:GetAttrs()
local first, last = attrs.COMPN_GlobalStart, attrs.COMPN_GlobalEnd
local flow = comp.CurrentFrame and comp.CurrentFrame.FlowView
local x, y = 0, 0
local taken = {{}}
if flow then
  x, y = flow:GetPos(node)
  for _, t in pairs(comp:GetToolList(false)) do
    local tx, ty = flow:GetPos(t)
    if tx then taken[#taken + 1] = {{tx, ty}} end
  end
  flow:Select()  -- a tool added beside a selected one is wired into its chain
end
{ROW_FOR_LUA}
local row = rowFor(x, y, #plates, taken)
if {str(dry_run).lower()} then
  for i, plate in ipairs(plates) do say("WOULD " .. plate[2] .. " " .. (x + 1 + i) .. "," .. row) end
  os.exit(0)
end
comp:Lock()
for i, plate in ipairs(plates) do
  local path, name = plate[1], plate[2]
  local loader = comp:FindTool(name)
  if not loader then
    loader = comp:AddTool("Loader", x + 1 + i, row)
    loader:SetAttrs({{TOOLS_Name = name}})
  end
  loader.Clip[fu.TIME_UNDEFINED] = comp:ReverseMapPath(path)
  loader.Loop[fu.TIME_UNDEFINED] = 1  -- one frame, repeated: the still holds across the comp
  loader.GlobalIn[fu.TIME_UNDEFINED] = first
  loader.GlobalOut[fu.TIME_UNDEFINED] = last
  say("LOADER " .. name)
end
comp:Unlock()
"""


def parse_output(text: str) -> list[str]:
    """The script's own lines. fuscript's banner and anything else are ignored."""
    return [line[len(PREFIX):].strip() for line in text.splitlines() if line.startswith(PREFIX)]


RESOLVE_SCRIPTING_OFF = ("Resolve's scripting is not reachable: set Preferences > System > General > "
                         "External scripting to Local")


def _run(fuscript: str, code: str, host: str = "fusion") -> tuple[list[str], str]:
    if not fuscript:
        return [], "the host's scripting tool was not found"
    try:
        done = subprocess.run([fuscript, "-l", "lua", "-x", code], capture_output=True, text=True,
                              timeout=TIMEOUT_SECONDS, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"scripting did not answer ({type(exc).__name__})"
    lines = parse_output(done.stdout)
    if "NOHOST" in lines:
        return lines, RESOLVE_SCRIPTING_OFF if host == "resolve" else "could not reach Fusion's scripting"
    if "NONODE" in lines:
        return lines, "could not find this node in an open comp"
    if not lines:
        tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or ["no output"]
        return lines, f"scripting failed: {tail[0][:120]}"
    return lines, ""


def loader_name(path: str) -> str:
    """The Loader takes the whole file name, prefix included, so stills with different prefixes on one
    frame get a Loader each. A tool name has letters, numbers and underscores, and starts with a letter."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(path))[0])
    return name if name[:1].isalpha() else f"LTX_{name}"


def requested_folder(host: str, folder: str) -> str:
    """Resolve's Fusion page has no comp file: Comp: there is a general folder in Documents, shared by
    every project. ProjectMedia: is Resolve's own per-project location, so a comp brought over from
    Fusion Studio saves there too, in its VFX folder."""
    if host == "resolve" and folder.startswith("Comp:"):
        return "ProjectMedia:/VFX/" + folder[len("Comp:"):].lstrip("/")
    return folder


def unreachable(folder: str, users: str = "/Users", volumes: str = "/Volumes") -> str:
    """Why a folder can never be made on this Mac, or "" if it can (or if only trying will tell).
    A project or comp set up on another Mac keeps that Mac's paths: a home folder of a user who isn't
    on this Mac, or a drive that isn't mounted. macOS refuses to make either, with a bare "Permission
    denied" or "No such file", which says nothing about the real cause."""
    folder = os.path.normpath(os.path.expanduser(folder))
    for root, what in ((users, "home folder"), (volumes, "drive")):
        prefix = os.path.normpath(root) + os.sep
        if not folder.startswith(prefix):
            continue
        top = folder[len(prefix):].split(os.sep)[0]
        if top and not os.path.isdir(os.path.join(root, top)):
            if what == "drive":
                return f"the drive {top} isn't mounted"
            return f"{os.path.join(root, top)} is the home folder of a user who isn't on this Mac"
    return ""


def fix_for(folder: str) -> str:
    """What to change, in the words of the setting the user can see."""
    if folder.startswith("ProjectMedia:"):
        return ("The project was probably set up on another Mac: change Project media location in Project "
                "Settings > General Options > Working Folders, or choose a Stills Folder")
    if folder.startswith("Comp:"):
        return "The comp was probably saved on another Mac: save it on this one, or choose a Stills Folder"
    return "Choose a Stills Folder on this Mac"


def choose_folder(requested: str, mapped: str, comp_saved: bool = True) -> tuple[str, bool]:
    """(folder to save in, True if it is the one asked for). Only a real absolute path counts, and Comp:
    only for a saved comp: otherwise it maps to a general folder, which looks like an answer but isn't."""
    if requested.startswith("Comp:") and not comp_saved:
        mapped = ""
    if mapped.startswith("/"):
        return mapped, True
    if requested.startswith("/") or requested.startswith("~"):
        return os.path.expanduser(requested), True
    return os.path.expanduser(FALLBACK_FOLDER), False


def locate(host: str, node_id: str, folder: str, fuscript: str | None = None) -> dict:
    """Before the job: where plates go, and the comp's name. Never raises."""
    fuscript = find_fuscript(host) if fuscript is None else fuscript
    folder = requested_folder(host, folder)
    lines, problem = _run(fuscript, locate_lua(host, node_id, folder), host)
    mapped = next((line[7:] for line in lines if line.startswith("MAPPED ")), "")
    comp_file = next((line[5:] for line in lines if line.startswith("COMP ")), "")
    clip = next((line[5:] for line in lines if line.startswith("CLIP ")), "")
    target, wanted = choose_folder(folder, mapped, bool(comp_file) or host == "resolve")
    stem = os.path.splitext(os.path.basename(comp_file or clip))[0]
    if not problem and not wanted:
        problem = f"{folder} has no location here" + (": save the comp first" if folder.startswith("Comp:") else "")
    why = unreachable(target) if wanted else ""
    if why:
        # Found now, before the slow part, so the stills still get saved and loaded, in the fallback folder.
        problem = f"{target} can't be made: {why}. {fix_for(folder)}"
        target, wanted = os.path.expanduser(FALLBACK_FOLDER), False
    return {"folder": target, "wanted": wanted, "comp_name": stem, "problem": problem, "fuscript": fuscript,
            "requested": folder}


def deliver(where: dict, host: str, node_id: str, paths: list[str], convert=None,  # noqa: ANN001
            why_no_convert: str = "") -> tuple[list[str], str]:
    """Moves each finished still from the cache to the chosen folder and adds Loaders. There is one
    file per still, which the node and the Loader both read. convert(source, target) writes it as
    DWAB; without it, or if it fails, the ZIP in the cache is what gets saved.
    Returns where the stills are now, and what to tell the user (empty if all went well)."""
    folder = where["folder"]
    notes = []
    saved = []
    if convert is None and why_no_convert:
        notes.append(f"Saved as ZIP, not DWAB: {why_no_convert}.")
    try:
        os.makedirs(folder, exist_ok=True)
        for path in paths:
            target = os.path.join(folder, os.path.basename(path))
            done = False
            if convert is not None:
                try:
                    convert(path, target)
                    done = True
                except Exception as exc:  # noqa: BLE001 - the still must not be lost over this
                    if not any(note.startswith("Saved as ZIP") for note in notes):
                        notes.append(f"Saved as ZIP, not DWAB: {str(exc)[:120]}.")
            if not done:
                shutil.copyfile(path, target + ".part")
                os.replace(target + ".part", target)
            saved.append(target)
    except OSError as exc:
        # Those saved so far stay as they are; the rest stay in the cache, where the node can read them.
        kept = saved + paths[len(saved):]
        for path in paths[:len(saved)]:
            os.unlink(path)
        why = unreachable(folder)
        reason = f"{why}. {fix_for(where.get('requested', ''))}" if why else str(exc)
        return kept, f"Could not save to {folder}: {reason}. Unsaved stills stay in the plugin's cache."
    for path in paths:
        os.unlink(path)
    if not where["wanted"]:
        notes.append(f"Saved to {folder} because {where['problem']}.")
    plates = [(p, loader_name(p)) for p in saved]
    lines, problem = _run(where["fuscript"], loaders_lua(host, node_id, plates), host)
    added = sum(1 for line in lines if line.startswith("LOADER "))
    if problem or added != len(plates):
        notes.append(f"Loaders not added: {problem or 'scripting stopped early'}.")
    return saved, " ".join(notes)
