# LTX Inpaint Still: changes against LTX Alpha Gen

LTX Inpaint Still makes clean plate stills in Fusion with the LTX in/outpainting IC-LoRA. It starts from the
CVT port of LTX Alpha Gen (`../cvtLTXAlphaGen/src`, cvt1.45), which is based on Jan Klier's Flame_LTXAlphaGen_OFX,
maintained by Logik.tv, under the Mozilla Public License 2.0. Those files stay under the MPL 2.0.

Tests that led to this design are in `../cvtLTXInpaintTest/ltx_inpaint_test.py`.

## Copied unchanged from Alpha Gen

`worker/ltx_worker.py`, `worker/ltxag_worker/` (`colour`, `fp16`, `log`, `padding`, `transport`, `oslayer`,
`mac_progress.swift`), `LICENSE`, `NOTICE`.

## New

- `worker/still_worker.py`: swaps the LoRA and runs one still job at a time, one-shot or resident. Its server is
  Alpha Gen's with a different job.
- `worker/ltxis_worker/fill.py`: the recipe. Detail (one stage) and Structure (two stages), Work Scale, mask grow,
  green reference, Laplacian blend. Two changes from LTX's workflow: each blend goes over the model's own fill
  instead of green, and the result fades back to the input's exact float values.
- `worker/ltxis_worker/exr.py`: a small ZIP half-float EXR writer, so no new dependency.
- `ofx/src/plugin.cpp`, `ofx/src/plate_io.h`: the node, far smaller than Alpha Gen's: no chunks, Frame Hold, Merge
  or Flame code.
- `worker/still_job.py`: one Process press: progress window, status file, saving.
- `worker/ltxis_worker/fusion_link.py`: path mapping and Loaders through the host's fuscript.
- `worker/ltxis_worker/prompt.py`, `person.py`, `person_check.swift`: the optional prompt's stored encoding, and the person check.
- `tools/still_cli.py`: runs a job from the command line, standing in for the plugin.
- `worker/tests/test_still.py`: tests without a model.
- `../macros/cvt_LTXOutpaintCanvas_vX.YY.setting`: a Fusion macro that sets up outpainting: the picture on a bigger
  canvas, over black, with the mask in its alpha (1 is the area to fill), so one wire goes to the yellow input and green
  stays empty. Moves are whole pixels and the chosen Filter only applies when Picture Size isn't 1, so the picture area
  stays exact.

## Versions

- **cvt1.29:** a Model Folder that was set up and has since been deleted is no longer offered from the saved settings: the field goes
  back to `~/LTX-2.5`. A folder on a drive that isn't mounted stays, since the drive may come back.

- **cvt1.28:** the folder fields have Choose buttons that open the Mac's Finder folder picker, in place of Fusion's browse
  control, which only picks files.

- **cvt1.27:** the viewer says SETUP REQUIRED until Setup is complete, as Alpha Gen's does. New Link Existing Models checkbox:
  Fusion never showed the question that used to ask, so Download always copied.

- **cvt1.26:** Setup's speed test for the extra 42 GB copy now needs 1.3x, not 1.2x, and doesn't decide again once either LTX-2.5
  plugin has decided, because an M5 Max sits right at 1.2x and could get the copy on one run and not the next.

- **cvt1.25:** the node has its own Setup, matching Alpha Gen's: licence, Hugging Face pages, token, Download
  (`helper/setup_helper.py`, `helper/model_manifest.json` with the in/outpainting LoRA, pinned by checksum). The Model Folder
  starts at `~/LTX-2.5`; anyone with Alpha Gen set up keeps its folder and only downloads the LoRA. New Collect Logs. Input
  Colour's first choice reads As Is (Rec.709, faster). Hardened: stills' names can't leave their folder, prompts stay out of
  the logs, a manifest survives any typed character, and archives unpack only inside their folder.

- **cvt1.24:** stills carry the fill's matte in alpha: the grown mask (Mask Grow), lossless in DWAB. The colour stays the whole
  picture, not premultiplied, and the plugin's Loaders aren't set to multiply it, so you see the full plate until you use the
  alpha. The node's own output has the same alpha.

- **cvt1.23:** Person Check is now Discard Stills with People, with clearer wording in its tooltip and Status. For ACEScg and
  Linear sRGB, the Copy for LLM picture and the check for people use OCIO's ACES 2.0 SDR view (its built-in studio config) instead of clipping. The check also uses person segmentation, which
  finds people seen from behind or out of focus: on a mask cutting through a real person, four stills in a row had an invented
  head that the face and body detections missed.

- **cvt1.22:** new Cache Folder in Setup, per user: stills in progress, encoded prompts and Copy for LLM pictures go there
  (`ofx/src/cache_folder.h`). It starts beside Alpha Gen's cache folder if that has been set. If the folder isn't available
  the home folder's cache is used and Status says so, and folders are never made on a volume that isn't mounted. The socket,
  logs and helpers stay in the home folder. New Input Colour: As Is, ACEScg or Linear sRGB. The two linear choices follow
  LTX's native HDR path (`worker/ltxis_worker/hdr.py`): the recipe runs on ACEScct values with the VAE at float32, the green
  stays its display value because that is what the LoRA was trained on, and stills come back in the input's space with values
  above 1.0 in the fill and the input's exact floats beyond the fade. Person Check and Copy for LLM look at a display version.

- **cvt1.21:** stills no longer fall back to `~/Documents/LTX Inpaint Stills` with no Loaders because a node could not find
  itself. Fusion can make a second instance of a node that changes its saved id, and the first kept using the old one, so
  the id is now read fresh each time.

- **cvt1.20:** new Copy for LLM button: puts instructions on the clipboard and shows the frame in Finder with the area to
  fill in grey, to paste into any LLM for a suggested prompt (`worker/still_llm.py`). The Prompt box wraps its text when you finish
  typing and scrolls top to bottom: OpenFX has no wrap setting, so the plugin puts in line breaks (`ofx/src/text_wrap.h`),
  which never reach the model. Fusion's control override was tried and rejected: it moves the control to the end of the
  page. Stills Folder, Prefix and Model Folder are one line tall. Loader rows are three apart, because tall tiles with long
  names still touched at two.

- **cvt1.19:** new optional Prompt: describe the whole scene as it should look. On big holes it keeps invented people and
  objects out (stairs: 16 of 16 clean, 1 of 8 with no prompt). Each new prompt is encoded once, in its own process, and
  kept in `prompts/` in the cache. New Person Check, on by default: macOS's own detection (`person_check.swift`, compiled
  on first use) looks at each still, and one with a person mostly in the fill is thrown away and replaced from the next
  seed, up to a limit. Status says what it threw out, and Seed moves past everything used.

- **cvt1.18:** Process no longer fails with kOfxStatFailed on a pasted node. Pasting left Reference Frame at 0 and marked
  as chosen, so it asked for a frame the comp doesn't have; a 0 like that now means the frame on screen. A Reference Frame
  outside the picture gets a clear message.

- **cvt1.17:** Loader rows are two apart, so batches don't overlap when tiles show their thumbnails.

- **cvt1.16:** growing the mask takes two quick passes with the same result, instead of one that took a minute or more at
  4K. Process stops with a message when the mask is larger than the picture, which put the fill on the wrong area.

- **cvt1.15:** Seed moves past the seeds just used after each finished job, so running again makes new stills instead of
  the same ones under new letters.

- **cvt1.14:** new Release Memory button unloads the model right away, for example before running LTX Alpha Gen. It
  refuses while a job is using the model.

- **cvt1.13:** a Prefix now goes before the clip's name instead of replacing it: `sign_sh010_LTXStill_f01032_A.exr`.
  The Inputs line moved into the folded Setup section.

- **cvt1.12:** Loaders no longer stack when the node has been dragged off the grid: the free row is found by measuring
  overlap with every tool, not by comparing whole-number grid cells.

- **cvt1.11:** Still Name is now Prefix, and it goes in the Loader's name too, as in `sign_LTXStill_f01032_A`. Loaders left
  it out, so a second prefix on one frame repointed the first prefix's Loader at the new file.

- **cvt1.10:** Show Variation down to Still Name move into their own Results section, apart from what goes into making
  a still.

- **cvt1.09:** when Process can't start, for example with no mask connected, a message window says why; it only went
  to the Status field before, which was easy to miss. The View option Still is now LTX AI Still.

- **cvt1.08:** new View setting. Original shows the picture at the frame the stills were made from, held on every
  frame, to compare with Still. It goes back to Still when a job finishes.

- **cvt1.07:** each new batch of Loaders goes in the first free row below the node; a second Process used to stack
  them exactly on the last ones. On Resolve's Fusion page Stills Folder defaults to
  `ProjectMedia:/VFX/ltx_inpaint_stills/`.

- **cvt1.06:** Method reads Detail (one pass) and Structure (two pass). The Inputs line reads "Yellow: image. Green:
  mask." Open Stills Folder now works before a job: it makes the folder stills will go in and shows it, and says so
  if that isn't the folder asked for (`worker/still_open_folder.py`).

- **cvt1.05:** Resolve's Fusion page: Stills Folder defaults to `ProjectMedia:/ltx_inpaint_stills/`, and a `Comp:` folder
  means the same there, because Comp: maps to a general folder in Documents when a comp has no file. Stills are named
  after the clip under the playhead. A never-saved comp in Fusion Studio no longer saves to that general folder without
  saying so. Status says how to turn on External scripting when Resolve can't be reached. New Open Stills Folder button.
  Tested on a test timeline: the Loader code finds the node, the Loader shows a DWAB and holds across frames.

- **cvt1.04:** stills are saved as DWAB at a new DWAB Level setting (default 45), and that is the only file per still:
  the node and the Loaders read the same one. OpenEXR 3.3.5 is built into the plugin from source (with Imath and
  libdeflate; BSD-3-Clause and MIT) to read it. The worker writes DWAB with the `openexr` Python package, installed on
  first use in the effect's own `pylib` folder, not the runtime it shares with Alpha Gen. If that can't be done, the
  still is saved as ZIP and Status says why; a still that can't be saved at all stays in the cache.

- **cvt1.03:** stills are saved with EXR's lossless ZIP compression. The node also still reads the uncompressed
  stills from earlier builds.

- **cvt1.02:** stills are named `sh060_LTXStill_f01032_A.exr` and saved to `Comp:/ltx_inpaint_stills/`. Letters carry on
  from what is already there for that name and frame, so a new Process never replaces earlier stills. Method and seed
  moved out of the name; they stay in the EXR's metadata. Loaders are named `LTXStill_f01032_A`. The node's wording says
  still, not plate.

- **cvt1.01:** plate names end in the variation's letter (`sh060_plate_f01032_detail_seed3_C.exr`), because Fusion
  loaded names ending in a number as one image sequence and only one variation showed. The inputs are labelled
  Image and Mask, and the Inspector says yellow is the image and green is the mask. Fusion assigns the colours.

- **cvt1.00 (in progress):** worker tested from the command line on four shots: Blockbuster box, hotel sign, stairs,
  diving board. First build of the node: picture and Mask inputs, Reference Frame, Method, Variations, Work Scale,
  held output of the chosen plate, EXRs saved to Plate Folder (`Comp:/plates/`), and Loaders added through
  Fusion's scripting (`worker/ltxis_worker/fusion_link.py`). The node and the Loader step have not been run in a
  host yet. Models and runtime are still set up by Alpha Gen; this effect's own Setup comes before release.
