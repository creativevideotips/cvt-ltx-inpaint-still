# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
"""Tests for the plate recipe and the EXR writer. No model needed: fake model passes stand in.

Run with the LTX venv's Python: python -B tests/test_still.py
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

import still_llm  # noqa: E402
import still_open_folder  # noqa: E402
import still_worker  # noqa: E402
from ltxis_worker import exr, fill, fusion_link, hdr, person, prompt  # noqa: E402


def solid_stage1(colour):
    def stage1(reference, _seed, _label):
        return np.broadcast_to(np.array(colour, np.float32), reference.shape).copy()

    return stage1


def echo_stage2(picture, _seed):
    return picture.copy()


class Blend(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.a = rng.random((192, 256, 3), dtype=np.float32)
        self.b = rng.random((192, 256, 3), dtype=np.float32)

    def test_black_mask_returns_the_base(self):
        out = fill.laplacian_blend(self.a, self.b, np.zeros((192, 256), np.float32), 0)
        self.assertLess(np.abs(out - self.b).max(), 1e-3)

    def test_white_mask_returns_the_fill(self):
        out = fill.laplacian_blend(self.a, self.b, np.ones((192, 256), np.float32), 0)
        self.assertLess(np.abs(out - self.a).max(), 1e-3)

    def test_small_picture_keeps_enough_pixels_for_the_blur(self):
        out = fill.laplacian_blend(self.a[:96, :96], self.b[:96, :96], np.zeros((96, 96), np.float32), 1)
        self.assertEqual(out.shape, (96, 96, 3))


class Plate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.picture = rng.random((270, 480, 3), dtype=np.float32)
        self.picture[10, 10] = (3.5, 0.25, -0.5)  # values a float picture may hold
        self.mask = np.zeros((270, 480), np.float32)
        self.mask[100:160, 200:280] = 1.0

    def make(self, method=fill.DETAIL, **kw):
        # A plain grey fill: the picture's noise averages about the same, so the blend's low bands agree.
        return fill.make_plate(self.picture, self.mask, method, 7, solid_stage1((0.5, 0.5, 0.5)), echo_stage2, **kw)

    def assert_filled_without_green(self, plate):
        inside = plate[110:150, 215:265]
        self.assertLess(inside.std(), self.picture[110:150, 215:265].std() * 0.5, "the picture's noise is still there")
        self.assertLess(abs(float((inside[..., 1] - inside[..., 0]).mean())), 0.02, "green came back")

    def test_detail_fills_and_keeps_exact_values_outside(self):
        plate = self.make(grow=4)
        self.assertEqual(plate.shape, self.picture.shape)
        self.assert_filled_without_green(plate)
        far = fill.grow_mask(self.mask, 4 + 2 + fill.fade_width(480, 270) + 2) == 0
        np.testing.assert_array_equal(plate[far], self.picture[far])
        np.testing.assert_array_equal(plate[10, 10], self.picture[10, 10])

    def test_the_fade_runs_from_solid_to_exactly_zero(self):
        mask = np.zeros((1, 400), np.float32)
        mask[0, 100] = 1.0
        fade = fill.fade_mask(mask, 10, 50)[0]
        self.assertEqual(fade[110], 1.0)
        self.assertEqual(fade[160], 0.0)
        self.assertTrue(np.all(np.diff(fade[100:200]) <= 0))
        self.assertLess(np.abs(np.diff(fade)).max(), 0.05, "a step would show as a line")

    def test_structure_and_work_scale_keep_the_size_and_outside(self):
        for method, scale in ((fill.STRUCTURE, 1), (fill.DETAIL, 2), (fill.STRUCTURE, 2)):
            with self.subTest(method=method, scale=scale):
                plate = self.make(method, grow=4, work_scale=scale)
                self.assertEqual(plate.shape, self.picture.shape)
                self.assert_filled_without_green(plate)
                far = fill.grow_mask(self.mask, 4 + 2 + fill.fade_width(480, 270) + 2) == 0
                np.testing.assert_array_equal(plate[far], self.picture[far])

    def test_growing_in_two_passes_matches_one_square_pass(self):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(6)
        mask = (rng.random((80, 120), dtype=np.float32) > 0.97).astype(np.float32)
        mask[30:40, 50:90] = 0.5
        for radius in (1, 4, 16):
            square = F.max_pool2d(torch.from_numpy(mask[None, None]), kernel_size=radius * 2 + 1, stride=1,
                                  padding=radius)[0, 0].numpy()
            np.testing.assert_array_equal(fill.grow_mask(mask, radius), square)

    def test_reference_has_green_under_the_grown_mask_and_is_on_the_grid(self):
        seen = {}

        def stage1(reference, _seed, _label):
            seen["reference"] = reference.copy()
            return reference

        fill.make_plate(self.picture, self.mask, fill.DETAIL, 7, stage1, echo_stage2, grow=4)
        h, w = seen["reference"].shape[:2]
        self.assertEqual((h % 32, w % 32), (0, 0))
        top = (h - 270) // 2
        np.testing.assert_allclose(seen["reference"][top + 130, 240], fill.GREEN)

    def test_the_callers_picture_is_not_changed(self):
        before = self.picture.copy()
        self.make(grow=4)
        np.testing.assert_array_equal(self.picture, before)

    def test_bad_jobs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            fill.check_job(self.picture, np.zeros((270, 480), np.float32), fill.DETAIL, 1)
        with self.assertRaisesRegex(ValueError, "too large"):
            fill.check_job(np.zeros((3240, 5760, 3), np.float32), np.ones((3240, 5760), np.float32), fill.DETAIL, 2)
        with self.assertRaisesRegex(ValueError, "picture is"):
            fill.check_job(self.picture, np.ones((10, 10), np.float32), fill.DETAIL, 1)

    def test_small_pictures_get_advice(self):
        small = np.zeros((183, 130, 3), np.float32)
        small_mask = np.ones((183, 130), np.float32)
        self.assertTrue(fill.check_job(small, small_mask, fill.DETAIL, 1))
        self.assertFalse(fill.check_job(small, small_mask, fill.DETAIL, 4))


class Names(unittest.TestCase):
    def test_the_prefix_goes_before_the_clip_name(self):
        self.assertEqual(fill.still_name("sign", "sh010"), "sign_sh010")
        self.assertEqual(fill.still_name("", "sh010"), "sh010")
        self.assertEqual(fill.still_name("sign", ""), "sign")  # scripting off: no clip name to be had
        self.assertEqual(fill.plate_file_name(fill.still_name("sign", "sh010"), 1032), "sign_sh010_LTXStill_f01032_A.exr")

    def test_frame_and_letter_are_in_the_name(self):
        self.assertEqual(fill.plate_file_name("sh060", 1032, 2), "sh060_LTXStill_f01032_C.exr")
        self.assertEqual(fill.still_name("sign", "sh010"), "sign_sh010")
        self.assertEqual(fill.still_name("", "../../etc/passwd"), "_.._etc_passwd")  # stays one name in its folder

    def test_no_name_ends_in_a_number(self):
        # Fusion would load names ending in a number as one image sequence.
        for variation in range(60):
            stem = fill.plate_file_name("sh060", 1032, variation)[:-len(".exr")]
            self.assertFalse(stem[-1].isdigit(), stem)

    def test_letters_go_past_z_and_back(self):
        self.assertEqual([fill.variation_letter(i) for i in (0, 25, 26, 27, 701, 702)], ["A", "Z", "AA", "AB", "ZZ", "AAA"])
        for i in range(800):
            self.assertEqual(fill.variation_number(fill.variation_letter(i)), i)

    def test_unsafe_characters_go(self):
        self.assertEqual(fill.plate_file_name("../my shot/", 5), "my_shot_LTXStill_f00005_A.exr")
        self.assertEqual(fill.plate_file_name("", 5), "plate_LTXStill_f00005_A.exr")

    def test_letters_carry_on_and_nothing_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(first), os.makedirs(second)
            self.assertEqual(fill.next_variation([first, second, os.path.join(tmp, "none")], "sh1", 7), 0)
            for folder, name in ((first, "sh1_LTXStill_f00007_A.exr"), (second, "sh1_LTXStill_f00007_C.exr"),
                                 (first, "sh1_LTXStill_f00008_F.exr"), (first, "other_LTXStill_f00007_H.exr")):
                with open(os.path.join(folder, name), "w") as f:
                    f.write("x")
            self.assertEqual(fill.next_variation([first, second], "sh1", 7), 3)  # after C: this name and frame only
            self.assertEqual(fill.next_variation([first, second], "sh1", 8), 6)
            self.assertEqual(fill.next_variation([first, second], "sh1", 9), 0)


class FusionLink(unittest.TestCase):
    """The parts that need no running host. The host's side is checked by hand in Fusion."""

    def test_only_a_real_path_counts_as_the_folder_asked_for(self):
        self.assertEqual(fusion_link.choose_folder("Comp:/plates/", "/Vol/shot/plates/"), ("/Vol/shot/plates/", True))
        self.assertEqual(fusion_link.choose_folder("/Vol/x/", ""), ("/Vol/x/", True))
        folder, wanted = fusion_link.choose_folder("Comp:/plates/", "Comp:/plates/")
        self.assertFalse(wanted)
        self.assertTrue(folder.endswith("LTX Inpaint Stills"))

    def test_comp_path_map_is_only_trusted_for_a_saved_comp(self):
        # An unsaved comp maps Comp: to a general folder in Documents: a real path, but the wrong one.
        folder, wanted = fusion_link.choose_folder("Comp:/stills/", "/home/x/Documents/Fusion/stills/", False)
        self.assertFalse(wanted)
        self.assertTrue(folder.endswith("LTX Inpaint Stills"))
        self.assertEqual(fusion_link.choose_folder("ProjectMedia:/stills/", "/Vol/p/stills/", False), ("/Vol/p/stills/", True))

    def test_comp_means_project_media_on_resolves_fusion_page(self):
        self.assertEqual(fusion_link.requested_folder("resolve", "Comp:/ltx_inpaint_stills/"),
                         "ProjectMedia:/VFX/ltx_inpaint_stills/")
        self.assertEqual(fusion_link.requested_folder("fusion", "Comp:/ltx_inpaint_stills/"), "Comp:/ltx_inpaint_stills/")
        self.assertEqual(fusion_link.requested_folder("resolve", "/Vol/x/"), "/Vol/x/")

    def test_locating_on_resolve_asks_for_the_clip_name(self):
        self.assertIn("GetCurrentVideoItem", fusion_link.locate_lua("resolve", "abc", "ProjectMedia:/s/"))
        self.assertNotIn("GetCurrentVideoItem", fusion_link.locate_lua("fusion", "abc", "Comp:/s/"))

    def test_the_folder_button_makes_the_folder_before_any_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            wanted = os.path.join(tmp, "stills")
            folder, note = still_open_folder.prepare("resolve", "abc", wanted, fuscript="")
            self.assertEqual((folder, note), (wanted, ""))
            self.assertTrue(os.path.isdir(wanted))

    @unittest.skipUnless(fusion_link.find_fuscript("fusion") or fusion_link.find_fuscript("resolve"), "needs fuscript")
    def test_a_new_batch_never_overlaps_tools_that_were_dragged_off_the_grid(self):
        fuscript = fusion_link.find_fuscript("fusion") or fusion_link.find_fuscript("resolve")

        def row(x, y, count, taken):
            table = ",".join(f"{{{a},{b}}}" for a, b in taken)
            code = fusion_link.ROW_FOR_LUA + f'io.write("LTXIS ", rowFor({x}, {y}, {count}, {{{table}}}), "\\n")'
            lines = fusion_link.parse_output(subprocess.run([fuscript, "-l", "lua", "-x", code], capture_output=True,
                                                            text=True, timeout=30).stdout)
            return float(lines[0])

        self.assertAlmostEqual(row(1, 1.009, 3, []), 4.009)
        # The stacked case: a first batch on the grid, then the node dragged to 1.506, 1.581.
        # A thumbnail tile is about 1.6 tall, so each row must clear the last by more than that.
        first_batch = [(3, 3.009), (4, 3.009), (5, 3.009)]
        placed = row(1.506, 1.581, 5, first_batch)
        self.assertGreaterEqual(placed - 3.009, 1.6)
        # A third batch clears the second one too.
        second_batch = [(1.506 + 1 + i, placed) for i in range(1, 6)]
        self.assertGreaterEqual(row(1.506, 1.581, 5, first_batch + second_batch) - placed, 1.6)
        # Tools off to the side don't push the batch down.
        self.assertAlmostEqual(row(1, 1, 3, [(20, 2)]), 4.0)

    def test_a_dry_run_of_the_loaders_adds_nothing(self):
        real = fusion_link.loaders_lua("fusion", "abc", [("/a/x.exr", "LTXStill_f00001_A")])
        self.assertIn("if false then", real)
        dry = fusion_link.loaders_lua("fusion", "abc", [("/a/x.exr", "LTXStill_f00001_A")], dry_run=True)
        self.assertIn("if true then", dry)
        self.assertLess(dry.index("os.exit(0)\nend\ncomp:Lock()"), dry.index('comp:AddTool("Loader"'))

    def test_each_prefix_gets_its_own_loader(self):
        sign = fusion_link.loader_name("/a/sign_LTXStill_f01032_A.exr")
        person = fusion_link.loader_name("/a/person_LTXStill_f01032_A.exr")
        self.assertEqual(sign, "sign_LTXStill_f01032_A")
        self.assertNotEqual(sign, person)

    def test_loader_names_are_valid_tool_names(self):
        self.assertEqual(fusion_link.loader_name("/a/sh.010-v2_LTXStill_f00001_B.exr"), "sh_010_v2_LTXStill_f00001_B")
        self.assertEqual(fusion_link.loader_name("/a/010_LTXStill_f00001_A.exr"), "LTX_010_LTXStill_f00001_A")

    def test_lua_strings_are_escaped(self):
        self.assertEqual(fusion_link.lua_string('a"b\\c'), '"a\\"b\\\\c"')

    def test_only_the_scripts_own_lines_are_read(self):
        text = "DaVinci Resolve Script Interpreter\nLTXIS MAPPED /a/b\nnoise\nLTXIS COMP /a/x.comp\n"
        self.assertEqual(fusion_link.parse_output(text), ["MAPPED /a/b", "COMP /a/x.comp"])

    def test_no_host_tool_still_saves_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "cache")
            os.makedirs(source)
            path = os.path.join(source, "sh1_LTXStill_f00007_A.exr")
            with open(path, "wb") as f:
                f.write(b"plate")
            where = {"folder": os.path.join(tmp, "saved"), "wanted": True, "comp_name": "", "problem": "", "fuscript": ""}
            paths, note = fusion_link.deliver(where, "fusion", "abc", [path])
            self.assertEqual(paths, [os.path.join(tmp, "saved", "sh1_LTXStill_f00007_A.exr")])
            self.assertTrue(os.path.exists(paths[0]))
            self.assertFalse(os.path.exists(path), "one file per still: nothing stays in the cache")
            self.assertIn("Loaders not added", note)

    def test_a_failed_conversion_still_saves_the_still(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sh1_LTXStill_f00007_A.exr")
            with open(path, "wb") as f:
                f.write(b"plate")

            def broken(_source, _target):
                raise RuntimeError("no DWAB today")

            where = {"folder": os.path.join(tmp, "saved"), "wanted": True, "comp_name": "", "problem": "", "fuscript": ""}
            paths, note = fusion_link.deliver(where, "fusion", "abc", [path], broken)
            with open(paths[0], "rb") as f:
                self.assertEqual(f.read(), b"plate")
            self.assertFalse(os.path.exists(path))
            self.assertIn("Saved as ZIP, not DWAB: no DWAB today", note)

    def test_no_library_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.exr")
            with open(path, "wb") as f:
                f.write(b"plate")
            where = {"folder": os.path.join(tmp, "saved"), "wanted": True, "comp_name": "", "problem": "", "fuscript": ""}
            _, note = fusion_link.deliver(where, "fusion", "abc", [path], None, "installing it failed")
            self.assertIn("Saved as ZIP, not DWAB: installing it failed", note)

    def test_a_folder_that_cannot_be_made_leaves_the_plates_where_they_are(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.exr")
            with open(path, "wb") as f:
                f.write(b"plate")
            blocker = os.path.join(tmp, "file")
            with open(blocker, "w") as f:
                f.write("x")
            where = {"folder": os.path.join(blocker, "sub"), "wanted": True, "comp_name": "", "problem": "", "fuscript": ""}
            paths, note = fusion_link.deliver(where, "fusion", "abc", [path])
            self.assertEqual(paths, [path])
            self.assertTrue(os.path.exists(path))
            self.assertIn("Could not save", note)


class Exr(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        rng = np.random.default_rng(3)
        self.rgb = (rng.random((40, 64, 3), dtype=np.float32) * 4.0 - 1.0).astype(np.float16).astype(np.float32)

    def test_values_and_metadata_come_back(self):
        path = os.path.join(self.dir, "a.exr")
        exr.write_exr(path, self.rgb, {"cvtPlateFrame": "1032"})
        back, meta = exr.read_exr(path)
        np.testing.assert_array_equal(back, self.rgb)
        self.assertEqual(meta["cvtPlateFrame"], "1032")

    def test_zip_is_lossless_on_noise_and_flat_areas_and_shrinks_flat_ones(self):
        path = os.path.join(self.dir, "b.exr")
        rng = np.random.default_rng(4)
        rgb = (rng.random((100, 120, 3), dtype=np.float32) * 5.0 - 1.0).astype(np.float16).astype(np.float32)
        rgb[40:70] = 0.25
        exr.write_exr(path, rgb)
        np.testing.assert_array_equal(exr.read_exr(path)[0], rgb)
        flat = os.path.join(self.dir, "c.exr")
        exr.write_exr(flat, np.full((100, 120, 3), 0.5, np.float32))
        self.assertLess(os.path.getsize(flat), 100 * 120 * 6 / 10)

    def test_uncompressed_files_from_the_first_builds_still_read(self):
        path = os.path.join(self.dir, "old.exr")
        h, w = self.rgb.shape[:2]
        header = exr._header(w, h, {}).replace(b"compression\0compression\0\x01\0\0\0\x03",
                                               b"compression\0compression\0\x01\0\0\0\0")
        head = struct.pack("<ii", exr.MAGIC, exr.VERSION) + header
        half = self.rgb.astype("<f2")
        with open(path, "wb") as f:
            f.write(head)
            np.arange(h, dtype="<u8").__mul__(8 + w * 6).__add__(len(head) + 8 * h).tofile(f)
            for y in range(h):
                f.write(struct.pack("<ii", y, w * 6))
                for _, c in exr.CHANNELS:
                    half[y, :, c].tofile(f)
        np.testing.assert_array_equal(exr.read_exr(path)[0], self.rgb)

    def test_dwab_is_much_smaller_close_and_keeps_its_metadata(self):
        library = os.path.expanduser("~/.cache/ltx-inpaint-still/pylib")
        uv = os.path.expanduser("~/LTX-2.5/runtime/bin/uv")
        openexr, why = exr.load_openexr(library, uv)
        if openexr is None:
            self.skipTest(f"no OpenEXR package: {why}")
        y, x = np.mgrid[0:256, 0:320].astype(np.float32)
        rng = np.random.default_rng(5)
        rgb = np.stack([x / 320, y / 256, (x + y) / 576], -1) + rng.random((256, 320, 3), dtype=np.float32) * 0.01
        zip_path, path = os.path.join(self.dir, "z.exr"), os.path.join(self.dir, "d.exr")
        exr.write_exr(zip_path, rgb)
        exr.write_dwab(openexr, path, rgb, 45, {"cvtPlateFrame": "1032"})
        # The matte stays exact in DWAB, soft edge included, and the colour under it is the whole picture.
        alpha = np.zeros(rgb.shape[:2], np.float32)
        alpha[8:24, 8:40] = 1.0
        alpha[8, 8:40] = 0.25
        with_alpha = os.path.join(self.dir, "alpha.exr")
        exr.write_dwab(openexr, with_alpha, rgb, 45, None, alpha=alpha)
        with openexr.File(with_alpha, separate_channels=True) as f:
            np.testing.assert_array_equal(f.channels()["A"].pixels.astype(np.float32), alpha)
        os.unlink(with_alpha)
        self.assertLess(os.path.getsize(path), os.path.getsize(zip_path))
        self.assertEqual(sorted(os.listdir(self.dir)), ["d.exr", "z.exr"], "no temporary file is left")
        with openexr.File(path) as f:
            self.assertEqual(f.header()["cvtPlateFrame"], "1032")
            channels = f.channels()
            back = channels["RGB"].pixels if "RGB" in channels else np.stack([channels[k].pixels for k in "RGB"], -1)
        self.assertLess(float(np.abs(back.astype(np.float32) - rgb).mean()), 0.01)

    def test_no_temporary_file_is_left(self):
        exr.write_exr(os.path.join(self.dir, "a.exr"), self.rgb)
        self.assertEqual(os.listdir(self.dir), ["a.exr"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_another_program_reads_it(self):
        path = os.path.join(self.dir, "a.exr")
        rgb = np.zeros((40, 64, 3), np.float32)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.75, 0.5, 0.25
        exr.write_exr(path, rgb)
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"],
                             capture_output=True, check=True).stdout
        values = np.frombuffer(out, "<f4").reshape(3, 40, 64)
        np.testing.assert_allclose(values[:, 20, 30], (0.5, 0.25, 0.75), atol=0.02)  # planes G, B, R


class StillAlpha(unittest.TestCase):
    def test_zip_keeps_alpha_and_colour(self):
        rgb = np.random.default_rng(1).random((40, 16, 3), np.float32) * 4.0
        alpha = np.linspace(0.0, 1.0, 40 * 16, dtype=np.float32).reshape(40, 16)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.exr")
            exr.write_exr(path, rgb, {"cvtPlateFrame": "7"}, alpha=alpha)
            back, back_alpha, meta = exr.read_exr_with_alpha(path)
            np.testing.assert_array_equal(back, rgb.astype(np.float16).astype(np.float32))
            np.testing.assert_array_equal(back_alpha, alpha.astype(np.float16).astype(np.float32))
            self.assertEqual(meta["cvtPlateFrame"], "7")
            exr.write_exr(path, rgb)
            self.assertIsNone(exr.read_exr_with_alpha(path)[1])


class LinearStills(unittest.TestCase):
    def test_acescct_known_values(self):
        # S-2016-001's own points: 0 is 0.0729, 1 is 0.5548, and the top of the range is about 222.9.
        out = hdr.to_acescct(np.array([[[0.0, 1.0, 222.86]]], np.float32), hdr.ACESCG)[0, 0]
        np.testing.assert_allclose(out, [0.0729055, 0.5547945, 1.0], atol=1e-4)

    def test_there_and_back(self):
        values = np.array([[[0.001, 0.18, 1.0], [2.5, 13.6, 100.0]]], np.float32)
        back = hdr.from_acescct(hdr.to_acescct(values, hdr.ACESCG), hdr.ACESCG)
        np.testing.assert_allclose(back, values, rtol=2e-3)

    def test_srgb_is_converted_through_acescg(self):
        grey = np.full((1, 1, 3), 0.18, np.float32)  # neutral stays neutral under both matrices
        np.testing.assert_allclose(hdr.from_acescct(hdr.to_acescct(grey, hdr.SRGB_LINEAR), hdr.SRGB_LINEAR), grey, rtol=2e-3)
        red = np.array([[[1.0, 0.0, 0.0]]], np.float32)
        self.assertNotAlmostEqual(float(hdr.to_acescct(red, hdr.SRGB_LINEAR)[0, 0, 1]), 0.0729, places=2)
        np.testing.assert_allclose(hdr.from_acescct(hdr.to_acescct(red, hdr.SRGB_LINEAR), hdr.SRGB_LINEAR), red, atol=2e-3)

    def test_the_matrices_undo_each_other(self):
        np.testing.assert_allclose(hdr.ACESCG_TO_SRGB @ hdr.SRGB_TO_ACESCG, np.eye(3), atol=1e-5)

    def plate(self, space):
        picture = np.full((256, 256, 3), 0.18, np.float32)
        picture[0, 0] = (-0.02, 300.0, 40.0)  # beyond ACEScct's range, far from the hole
        picture[150, 40] = (5.0, 5.0, 5.0)
        mask = np.zeros((256, 256), np.float32)
        mask[100:140, 100:140] = 1.0
        return picture, mask, hdr.make_plate(picture, mask, space, fill.DETAIL, 1, solid_stage1((0.6, 0.6, 0.6)),
                                             lambda p, _s: p, grow=4)

    def test_exact_values_come_back_outside_the_fade(self):
        for space in hdr.LINEAR_SPACES:
            picture, _mask, plate = self.plate(space)
            self.assertTrue(np.array_equal(plate[0, 0], picture[0, 0]))

    def test_the_fill_is_scene_linear(self):
        # Most of the picture is the hole, so the blend leaves the fill's own level: 0.6 in ACEScct is
        # about 1.7 in ACEScg, so a fill can hold values above 1.0.
        picture = np.full((256, 256, 3), 0.18, np.float32)
        mask = np.ones((256, 256), np.float32)
        mask[:12] = 0.0
        plate = hdr.make_plate(picture, mask, hdr.ACESCG, fill.DETAIL, 1, solid_stage1((0.6, 0.6, 0.6)), lambda p, _s: p, grow=4)
        self.assertGreater(float(plate[200, 128, 1]), 1.0)

    def test_display_is_for_looking_only(self):
        out = hdr.to_display(np.array([[[0.18, 0.18, 0.18], [8.0, 8.0, 8.0]]], np.float32), hdr.ACESCG)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.349, places=2)  # the ACES 2.0 SDR view's mid grey
        self.assertLess(float(out[0, 1, 0]), 1.0)
        for space in hdr.LINEAR_SPACES:  # OCIO and the fallback both keep brighter looking brighter
            values = np.array([[[2.0] * 3, [4.0] * 3, [14.0] * 3]], np.float32)
            for top in (hdr.to_display(values, space)[0, :, 0], hdr._rolled_off(values, space)[0, :, 0]):
                self.assertTrue(top[0] < top[1] < top[2], top)
        same = np.array([[[0.25, 0.5, 2.0]]], np.float32)
        np.testing.assert_allclose(hdr.to_display(same, hdr.AS_IS), [[[0.25, 0.5, 1.0]]])


class CopyForLlm(unittest.TestCase):
    def test_only_the_fill_goes_grey(self):
        picture = np.full((4, 4, 3), 0.2, np.float32)
        mask = np.zeros((4, 4), np.float32)
        mask[0, 0] = 1.0
        pixels = np.asarray(still_llm.blanked(picture, mask))
        self.assertEqual(pixels[0, 0].tolist(), [128, 128, 128])
        self.assertEqual(pixels[3, 3].tolist(), [51, 51, 51])

    def test_a_big_frame_gets_smaller_and_keeps_its_shape(self):
        image = still_llm.blanked(np.zeros((1920, 1080, 3), np.float32), np.zeros((1920, 1080), np.float32))
        self.assertEqual(image.size, (720, 1280))

    def test_the_instructions_follow_the_prompt_rules(self):
        self.assertIn("one sentence", still_llm.INSTRUCTIONS)
        self.assertIn("Never mention people", still_llm.INSTRUCTIONS)


class PersonCheck(unittest.TestCase):
    def box(self, kind, x, y, w, h):
        return {"kind": kind, "x": x, "y": y, "w": w, "h": h, "confidence": 0.9}

    def test_counts_only_what_lies_mostly_in_the_fill(self):
        mask = np.zeros((100, 100), np.float32)
        mask[:50] = 1.0  # the top half is the fill
        found = person.people_in_fill([self.box("face", 0.1, 0.1, 0.2, 0.2),  # all in the fill
                                       self.box("body", 0.1, 0.4, 0.2, 0.5),  # a fifth in the fill, like an outpaint's feet
                                       self.box("upper", 0.5, 0.25, 0.2, 0.5)], mask)  # exactly half
        self.assertEqual(found, ["face", "upper"])

    def grid(self, cells):
        grid = np.zeros((person.GRID_ROWS, person.GRID_COLUMNS), np.uint8) + ord("0")
        grid[cells] = ord("1")
        return grid.tobytes().decode()

    def test_a_person_seen_from_behind_counts_by_area(self):
        mask = np.zeros((1080, 1920), np.float32)
        mask[:, :960] = 1.0  # the left half is the fill
        none_kept = self.grid((slice(0, 20), slice(0, 20)))
        self.assertTrue(person.person_area_in_fill(none_kept, mask))
        finished = self.grid((slice(0, 40), slice(70, 130)))  # as much in the fill as out of it
        self.assertTrue(person.person_area_in_fill(finished, mask))

    def test_an_outpaint_keeps_a_real_persons_legs(self):
        mask = np.zeros((1080, 1920), np.float32)
        mask[:, :960] = 1.0
        self.assertFalse(person.person_area_in_fill(self.grid((slice(0, 40), slice(90, 150))), mask))  # 6 in, 54 out

    def test_a_speck_or_a_bad_grid_does_not_count(self):
        mask = np.ones((1080, 1920), np.float32)
        self.assertFalse(person.person_area_in_fill(self.grid((slice(0, 3), slice(0, 3))), mask))
        self.assertFalse(person.person_area_in_fill("", mask))

    def test_a_box_off_the_picture_does_not_crash(self):
        mask = np.ones((10, 10), np.float32)
        self.assertEqual(person.people_in_fill([self.box("body", -0.5, -0.5, 0.2, 0.2)], mask), [])

    def test_png_reads_back(self):
        from PIL import Image

        picture = np.array([[[0.0, 0.5, 1.0], [2.0, -1.0, 0.25]]], np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.png")
            person.write_png(path, picture)
            with Image.open(path) as im:
                self.assertEqual(np.asarray(im.convert("RGB")).tolist(), [[[0, 128, 255], [255, 0, 64]]])

    def test_a_missing_helper_says_why_and_finds_nothing(self):
        kinds, why = person.check("/no/such/helper", np.zeros((4, 4, 3), np.float32), np.ones((4, 4), np.float32))
        self.assertEqual(kinds, [])
        self.assertTrue(why)


class MakeStills(unittest.TestCase):
    def run_it(self, seeds, spare, people, problem=""):
        saved, lines = [], []

        def find_people(plate):
            return ([], problem) if problem else ((["face"] if plate in people else []), "")

        next_seed = still_worker.make_stills(seeds, spare, lambda seed: seed, find_people,
                                             lambda plate, seed, n, _s: saved.append((seed, n)),
                                             lambda *parts: lines.append(parts))
        return saved, [line[1] for line in lines if line[0] == "rejected"], next_seed

    def test_nothing_found(self):
        self.assertEqual(self.run_it([1, 2], 3, set()), ([(1, 0), (2, 1)], [], 3))

    def test_a_replacement_takes_the_next_letter_and_seed(self):
        self.assertEqual(self.run_it([1, 2], 3, {1}), ([(2, 0), (3, 1)], [1], 4))

    def test_stops_at_the_limit(self):
        saved, rejected, next_seed = self.run_it([1, 2], 2, {1, 2, 3, 4, 5})
        self.assertEqual((saved, rejected, next_seed), ([], [1, 2, 3, 4], 5))

    def test_a_check_that_fails_keeps_the_stills(self):
        saved, rejected, _ = self.run_it([1, 2], 3, {1}, problem="the detector stopped")
        self.assertEqual((saved, rejected), ([(1, 0), (2, 1)], []))


class Prompt(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(prompt.clean("  a   wooden\nstaircase\r\n"), "a wooden staircase")
        self.assertEqual(prompt.clean(None), "")

    def test_path_follows_text_and_encoder(self):
        with tempfile.TemporaryDirectory() as model:
            encoder = os.path.join(model, prompt.TEXT_ENCODER)
            os.makedirs(os.path.dirname(encoder))
            with open(encoder, "wb") as f:
                f.write(b"12")
            a = prompt.embedding_path("/c", "stairs", model)
            self.assertEqual(a, prompt.embedding_path("/c", "stairs", model))
            self.assertNotEqual(a, prompt.embedding_path("/c", "pool", model))
            with open(encoder, "wb") as f:
                f.write(b"123")
            self.assertNotEqual(a, prompt.embedding_path("/c", "stairs", model))


if __name__ == "__main__":
    unittest.main(verbosity=1)
