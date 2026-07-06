#!/usr/bin/env python
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from common.sa_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--attr", "GENDER", "--model", "llama_3_1_8b_instant", *sys.argv[1:]]))
