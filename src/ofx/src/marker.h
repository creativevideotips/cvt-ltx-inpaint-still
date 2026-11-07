// Marker drawn into the matte when a frame has no result (decision 40): a centred caption
// such as NOT PROCESSED. cvt1.21: full strength; cvt1.22: no diagonal stripes, bolder letters.
// Built-in 5x7 font covering only the letters the captions need. Pure, no OFX.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ltxag {

constexpr float kMarkerText = 1.0f;

// matte: w*h floats, rows top-down, overwritten. Unsupported characters render as spaces.
void drawMarker(std::vector<float>& matte, int w, int h, const std::string& caption);

// Decision 58: the image output shows the source with the matte overlaid in magenta at 30%
// (50% was too opaque, user after test 4): out = src * (1 - 0.3 m) + magenta * 0.3 m, per RGB
// channel. m = 0 leaves the source as is.
constexpr float kOverlayColour[3] = {1.0f, 0.0f, 1.0f};
constexpr float kOverlayOpacity = 0.3f;
inline void overlayPixel(const float* src, float m, float* out) {
  const float o = kOverlayOpacity * m;
  for (int k = 0; k < 3; ++k) out[k] = src[k] * (1.0f - o) + kOverlayColour[k] * o;
}

// Decision 76: in Input Colour Auto the picture arrives as ACEScg. The image output is an illustration,
// so it goes out display-referred (tagged Rec.1886 Rec.709 - Display) and the overlay is blended after
// this conversion, so it looks the same whatever the clip's own colourspace. AP1 to Rec.709 (D60 to D65,
// Bradford), gamma 2.4, clamped to 0..1: no tone mapping, as Flame's own display conversion. Checked
// against Flame's aces2.0 config (OCIO ACEScg -> Rec.1886 Rec.709 - Display, 9 colours, 4 decimals).
inline void acescgToDisplay(const float* in, float* out) {
  static const float M[3][3] = {{1.70505f, -0.62179f, -0.08326f},
                                {-0.13026f, 1.14080f, -0.01055f},
                                {-0.02400f, -0.12897f, 1.15297f}};
  for (int k = 0; k < 3; ++k) {
    const float lin = M[k][0] * in[0] + M[k][1] * in[1] + M[k][2] * in[2];
    out[k] = std::pow(std::min(std::max(lin, 0.0f), 1.0f), 1.0f / 2.4f);
  }
}

// cvt1.26: Alpha Black Clip / Alpha White Clip. The model's decoder leaves faint noise in the
// matte (about 0.001-0.005 where it should be 0, 0.99-0.999 where it should be 1). Values at or
// below `black` become 0, at or above `white` 1, and the ones between are stretched to fill
// 0..1, so soft edges stay soft. black 0 and white 1 leave the matte as it is.
inline float clipMatteValue(float v, float black, float white) {
  if (white <= black) return v > black ? 1.0f : 0.0f;
  return std::min(1.0f, std::max(0.0f, (v - black) / (white - black)));
}
inline void clipMatte(std::vector<float>& matte, float black, float white) {
  if (black <= 0.0f && white >= 1.0f) return;
  for (float& v : matte) v = clipMatteValue(v, black, white);
}

}  // namespace ltxag
