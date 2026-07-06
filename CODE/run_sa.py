#!/usr/bin/env python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.sa_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
