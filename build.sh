#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cvtLTXInpaint build.sh
#
# From Creative Video Tips (CVT) - LTX Inpaint Still for Fusion Studio and
# DaVinci Resolve's Fusion page, based on the CVT port of LTX Alpha Gen
# (MPL 2.0, see src/LICENSE and src/CVT_PORT.md).
#
# What it does:
#   1. Builds the OFX bundle into build/LTXInpaintStill.ofx.bundle.
#   2. Runs the tests, including the worker's, in the LTX runtime's Python.
#
#   ./build.sh
#
# Nothing is installed: run ./install.sh afterwards.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build"
PORT_VERSION="cvt1.30"   # the version shown on the node, and (as 1.00) the plugin version hosts store
LTX_PYTHON="$HOME/LTX-2.5/runtime/ltx2/.venv/bin/python"

command -v cmake >/dev/null || { echo "cmake is missing: brew install cmake" >&2; exit 1; }

echo "== building $PORT_VERSION"
cmake -S "$HERE/src" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLTXIS_VERSION="${PORT_VERSION#cvt}" \
  -DWORKER_TEST_PYTHON="$LTX_PYTHON" >/dev/null
cmake --build "$BUILD" -j "$(sysctl -n hw.ncpu)" 2>&1 | grep -E "error|warning:" || true
[ -f "$BUILD/LTXInpaintStill.ofx.bundle/Contents/MacOS/LTXInpaintStill.ofx" ] || { echo "build FAILED" >&2; exit 1; }

echo "== tests"
(cd "$BUILD" && ctest --output-on-failure | tail -3)
if [ -x "$LTX_PYTHON" ]; then
  (cd "$HERE/src/worker" && "$LTX_PYTHON" -B tests/test_still.py 2>&1 | grep -E '^(Ran|OK|FAILED|FAIL:|ERROR:)')
else
  echo "(worker tests skipped: no LTX runtime at $LTX_PYTHON)"
fi
echo "== done: $BUILD/LTXInpaintStill.ofx.bundle"
