#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cvtLTXInpaint install.sh
#
# From Creative Video Tips (CVT) - LTX Inpaint Still for Fusion Studio and
# DaVinci Resolve's Fusion page, based on the CVT port of LTX Alpha Gen
# (MPL 2.0, see src/LICENSE and src/CVT_PORT.md).
#
# What it does:
#   1. Copies build/LTXInpaintStill.ofx.bundle into /Library/OFX/Plugins, where
#      Resolve and Fusion Studio look for OFX plugins (asks for your password: sudo).
#   2. Removes the quarantine flag so macOS lets them load it.
#
#   ./install.sh              install (replaces an earlier copy)
#   ./install.sh --uninstall  remove it
#
# Quit Resolve and Fusion first; they find the plugin at their next start.
# Models and the LTX runtime are not touched.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=LTXInpaintStill.ofx.bundle
DEST=/Library/OFX/Plugins

if pgrep -x Resolve >/dev/null || pgrep -x Fusion >/dev/null; then
  echo "Resolve or Fusion is running: quit it first, then run this again. Nothing was changed." >&2
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then exec sudo "$0" "$@"; fi

if [ "${1:-}" = "--uninstall" ]; then
  rm -rf "${DEST:?}/$NAME"
  echo "Removed $DEST/$NAME."
  exit 0
fi

SRC="$HERE/build/$NAME"
[ -f "$SRC/Contents/MacOS/LTXInpaintStill.ofx" ] || { echo "No build found: run ./build.sh first." >&2; exit 1; }
mkdir -p "$DEST"
rm -rf "${DEST:?}/$NAME"
ditto "$SRC" "$DEST/$NAME"
xattr -dr com.apple.quarantine "$DEST/$NAME" 2>/dev/null || true
chmod -R go-w "$DEST/$NAME"
echo "Installed $DEST/$NAME. In Fusion: Effects > OpenFX > Creative Video Tips > LTX Inpaint Still."
