// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// The plugin must read back what the worker saves: ZIP exactly, and DWAB closely.
//   test_plate_io <python> <worker dir>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "cache_folder.h"
#include "plate_io.h"
#include "text_wrap.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  // The Cache Folder: its default, and that a missing drive is never created.
  std::string note;
  const std::string root = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp/") + "/ltxis_cache_test";
  if (defaultCacheFolder("/Volumes/drive/Fusion/LTXAlphaGen_Cache/", "/home") != "/Volumes/drive/Fusion/LTXInpaint_Cache" ||
      defaultCacheFolder("", "/home") != "/home" || chooseCacheFolder("", "/home", note) != "/home" || !note.empty() ||
      chooseCacheFolder(root + "/a/b", "/home", note) != root + "/a/b" || !note.empty()) {
    std::printf("cache folder choice failed\n");
    return 1;
  }
  struct stat gone {};
  if (chooseCacheFolder("/Volumes/ltxis_no_such_volume/cache", "/home", note) != "/home" || note.empty() ||
      stat("/Volumes/ltxis_no_such_volume", &gone) == 0 || chooseCacheFolder("relative/path", "/home", note) != "/home") {
    std::printf("missing drive handling failed: %s\n", note.c_str());
    return 1;
  }
  rmdir((root + "/a/b").c_str());
  rmdir((root + "/a").c_str());
  rmdir(root.c_str());
  // The Prompt box's wrapping: words stay whole, it settles, and the words themselves never change.
  const std::string once = wrapText("  a wooden\n staircase in   a bright home ", 12);
  if (once != "a wooden\nstaircase in\na bright\nhome" || wrapText(once, 12) != once || wrapText("", 12) != "" ||
      wrapText("incomprehensible word", 5) != "incomprehensible\nword") {
    std::printf("wrapText gave: %s\n", once.c_str());
    return 1;
  }
  const std::string base = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/ltxis_test_plate";
  const std::string zipPath = base + ".exr", dwabPath = base + "_dwab.exr", alphaPath = base + "_alpha.exr";
  std::remove(dwabPath.c_str());
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  const std::string code =
      "import sys, os; sys.path.insert(0, '" + std::string(argv[2]) + "'); import numpy as np; "
      "from ltxis_worker import exr; a = np.zeros((40, 4, 3), np.float32); a[0, 1] = (0.25, 0.5, 2.5); "
      "b = np.zeros((64, 128), np.float32); b[10:20, 30:60] = 1.0; b[10, 30] = 0.5; "
      "a[2, 3] = (-1.0, 1.0, 0.125); a[39, 0] = (0.75, 0.0, 3.0); "  // past the first 16-line block
      "exr.write_exr('" + zipPath + "', a, {'cvtPlateFrame': '7'}); "
      "m, why = exr.load_openexr('" + home + "/.cache/ltx-inpaint-still/pylib', ''); "
      "m and exr.write_dwab(m, '" + dwabPath + "', np.full((64, 128, 3), (0.25, 0.5, 0.75), np.float32), 45, alpha=b); "
      "exr.write_exr('" + alphaPath + "', a, alpha=np.tile(np.linspace(0, 1, 40, dtype=np.float32)[:, None], (1, 4)))";
  if (std::system((std::string(argv[1]) + " -B -c \"" + code + "\"").c_str()) != 0) return 2;
  ltxis::Plate plate;
  std::string err;
  int bad = 0;
  if (!ltxis::readPlate(zipPath, plate, err)) {
    std::printf("ZIP read failed: %s\n", err.c_str());
    return 1;
  }
  const auto check = [&](const ltxis::Plate& p, int y, int x, float r, float g, float b, float tolerance) {
    const float* v = p.rgb.data() + (size_t(y) * p.width + x) * 3;
    if (std::fabs(v[0] - r) > tolerance || std::fabs(v[1] - g) > tolerance || std::fabs(v[2] - b) > tolerance) {
      std::printf("pixel %d,%d is %g %g %g\n", y, x, v[0], v[1], v[2]);
      ++bad;
    }
  };
  if (plate.width != 4 || plate.height != 40) ++bad;
  check(plate, 0, 1, 0.25f, 0.5f, 2.5f, 1e-3f);
  check(plate, 2, 3, -1.0f, 1.0f, 0.125f, 1e-3f);
  check(plate, 39, 0, 0.75f, 0.0f, 3.0f, 1e-3f);
  check(plate, 1, 1, 0.0f, 0.0f, 0.0f, 1e-3f);
  // Stills from before alpha was added read as solid; a saved alpha comes through, with colour left alone.
  if (plate.alpha[0] != 1.0f) ++bad;
  ltxis::Plate zipAlpha;
  if (!ltxis::readPlate(alphaPath, zipAlpha, err) || std::fabs(zipAlpha.alpha[4 * 39] - 1.0f) > 1e-3f ||
      std::fabs(zipAlpha.alpha[0]) > 1e-3f) {
    std::printf("ZIP alpha failed\n");
    ++bad;
  }
  check(zipAlpha, 0, 1, 0.25f, 0.5f, 2.5f, 1e-3f);
  ltxis::Plate none;
  if (ltxis::readPlate(zipPath + ".missing", none, err)) ++bad;
  ltxis::Plate dwab;
  if (std::ifstream(dwabPath).good()) {
    if (!ltxis::readPlate(dwabPath, dwab, err)) {
      std::printf("DWAB read failed: %s\n", err.c_str());
      ++bad;
    } else {
      if (dwab.width != 128 || dwab.height != 64) ++bad;
      check(dwab, 30, 60, 0.25f, 0.5f, 0.75f, 0.02f);
    }
  } else {
    std::printf("(no DWAB check: the OpenEXR package is not installed)\n");
  }
  std::remove(zipPath.c_str());
  std::remove(dwabPath.c_str());
  std::printf(bad ? "FAILED\n" : "ok\n");
  return bad ? 1 : 0;
}
