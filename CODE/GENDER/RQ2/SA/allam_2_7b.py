#!/usr/bin/env python
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from common.sa_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--attr", "GENDER", "--model", "allam_2_7b", *sys.argv[1:]]))
