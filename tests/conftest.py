"""Unit tests for the parts of Venno that need no model (python -m pytest).

The engine reads its options from the command line when it is imported: run it on the CPU here, whatever the
machine has, so the tests behave the same everywhere.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], "--cpu"]
