# Pytest setup shared by the whole test suite.
#
# The project's modules live in src/ and import each other by bare name
# (e.g. "from features import ..."), and the pipeline scripts use
# project-root-relative data paths (e.g. "data/processed/..."). So the tests
# need two things:
#   1. src/ on Python's import path, so "import calibration" etc. resolve.
#   2. To be run from the project root, so the relative data paths point at
#      the real frozen artefacts. Run "py -m pytest" from the Artefact folder.
#
# pytest auto-loads this file before collecting any tests, so the path
# insertion below happens once for every test module without me having to
# repeat it anywhere.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # two levels up from this file = project root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)  # prepend so src/ modules take priority over any same-named installed packages
