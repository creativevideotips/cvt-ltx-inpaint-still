// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// Reads a still's EXR. No OpenEXR headers here: OpenEXR needs C++17 and the OFX support library cannot
// compile under it, so the reading is compiled on its own in plate_io.cpp.
#pragma once

#include <string>
#include <vector>

namespace ltxis {

struct Plate {
  int width = 0, height = 0;
  std::vector<float> rgb;  // width * height * 3, rows top-down; the whole picture, not premultiplied
  std::vector<float> alpha;  // width * height: the fill's matte, or all 1 in stills from before it was added
  bool valid() const { return width > 0 && height > 0 && !rgb.empty(); }
};

// DWAB as saved, or the ZIP or uncompressed stills of a fallback or an earlier build.
bool readPlate(const std::string& path, Plate& out, std::string& err);

}  // namespace ltxis
