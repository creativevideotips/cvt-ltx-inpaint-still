// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// The Cache Folder setting: which folder to use, and what to do when it isn't there. The plugin's
// socket, logs and helpers stay in the home folder whatever this says: sockets may not work on a
// network drive, and those are needed even when the drive isn't mounted.
#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <string>

// Beside Alpha Gen's cache folder if it has one, so both caches are in one place with no setup.
inline std::string defaultCacheFolder(std::string alphaGenCache, const std::string& fallback) {
  while (alphaGenCache.size() > 1 && alphaGenCache.back() == '/') alphaGenCache.pop_back();
  const size_t slash = alphaGenCache.rfind('/');
  if (alphaGenCache.empty() || slash == std::string::npos || slash == 0) return fallback;
  return alphaGenCache.substr(0, slash) + "/LTXInpaint_Cache";
}

// Making folders under /Volumes while a drive is away would leave a local folder in the drive's
// place, which then hides the drive when it comes back. So a missing volume is never created.
inline bool volumeIsMissing(const std::string& path) {
  const std::string root = "/Volumes/";
  if (path.rfind(root, 0) != 0) return false;
  const size_t end = path.find('/', root.size());
  struct stat st {};
  return stat(path.substr(0, end).c_str(), &st) != 0;
}

inline bool makeFolders(const std::string& path) {
  for (size_t at = path.find('/', 1); at != std::string::npos; at = path.find('/', at + 1))
    mkdir(path.substr(0, at).c_str(), 0755);
  mkdir(path.c_str(), 0755);
  struct stat st {};
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && access(path.c_str(), W_OK) == 0;
}

// The folder to use now. If the chosen one can't be used, the fallback, and `note` says so.
inline std::string chooseCacheFolder(const std::string& chosen, const std::string& fallback, std::string& note) {
  note.clear();
  if (chosen.empty()) return fallback;
  if (chosen[0] == '/' && !volumeIsMissing(chosen) && makeFolders(chosen)) return chosen;
  note = "Cache Folder " + chosen + " is not available, so the home folder's cache was used.";
  return fallback;
}
