// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
#include "plate_io.h"

#include <ImathBox.h>
#include <ImfArray.h>
#include <ImfRgbaFile.h>

#include <exception>

namespace ltxis {

bool readPlate(const std::string& path, Plate& out, std::string& err) {
  try {
    Imf::RgbaInputFile file(path.c_str());
    const Imath::Box2i window = file.dataWindow();
    const int width = window.max.x - window.min.x + 1, height = window.max.y - window.min.y + 1;
    if (width <= 0 || height <= 0) {
      err = "the still is empty: " + path;
      return false;
    }
    Imf::Array2D<Imf::Rgba> pixels(height, width);
    file.setFrameBuffer(&pixels[0][0] - window.min.x - window.min.y * width, 1, size_t(width));
    file.readPixels(window.min.y, window.max.y);
    out.width = width, out.height = height;
    out.rgb.resize(size_t(width) * height * 3);
    out.alpha.resize(size_t(width) * height);
    float* q = out.rgb.data();
    float* a = out.alpha.data();
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const Imf::Rgba& p = pixels[y][x];
        *q++ = float(p.r), *q++ = float(p.g), *q++ = float(p.b);
        *a++ = float(p.a);  // OpenEXR fills in 1 for a file with no alpha
      }
    }
    return true;
  } catch (const std::exception& e) {
    err = "cannot read " + path + ": " + e.what();
    return false;
  }
}

}  // namespace ltxis
