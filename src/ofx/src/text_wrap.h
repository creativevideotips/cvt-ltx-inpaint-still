// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
#pragma once

#include <sstream>
#include <string>

// The text's words on lines of at most `columns` characters, joined by single spaces and line breaks.
// A word longer than a line keeps a line of its own. Wrapping wrapped text gives the same text back.
inline std::string wrapText(const std::string& text, size_t columns) {
  std::istringstream in(text);
  std::string word, out;
  size_t line = 0;
  while (in >> word) {
    if (line > 0 && line + 1 + word.size() > columns) {
      out += '\n';
      line = 0;
    } else if (line > 0) {
      out += ' ';
      ++line;
    }
    out += word;
    line += word.size();
  }
  return out;
}
