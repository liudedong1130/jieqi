#!/usr/bin/env python3
"""Build the vendored C++ Musesfish query binary without requiring CMake."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "agents" / "vendor" / "musesfish_cpp"
    out_dir = src / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "musesfish_query"

    sources = (
        sorted((src / "global").glob("*.cpp"))
        + sorted((src / "score").glob("*.cpp"))
        + sorted((src / "board").glob("*.cpp"))
        + [src / "query_main.cpp"]
    )
    cmd = ["clang++", "-std=c++17", "-O2", "-DNDEBUG", *map(str, sources), "-o", str(out)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"built: {out}")


if __name__ == "__main__":
    main()
