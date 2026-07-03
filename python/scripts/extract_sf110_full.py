#!/usr/bin/env python3
"""
Robust SF110 full-archive extractor (Windows long-path safe).

The SF110 zip contains 111 projects with very deep EvoSuite package paths that
exceed the Windows MAX_PATH (260 char) limit. Standard extraction crashes on the
first offending file. This extractor:
  - uses the Windows extended-length path prefix (\\?\) to bypass MAX_PATH;
  - tolerates per-file failures (logs them, keeps going);
  - reports how many projects end up usable (build.xml + src/main/java).

Usage:
    python python/scripts/extract_sf110_full.py
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

ZIP = Path("python/data/raw/evosuite/sf110/SF110-20130704-src.zip")
TARGET = Path("python/data/raw/evosuite/sf110/extracted")
EXT_PREFIX = "\\\\?\\"  # literal \\?\


def long_path(p: str) -> str:
    """Return an extended-length absolute path on Windows to bypass MAX_PATH."""
    ap = os.path.abspath(p)
    if os.name == "nt" and not ap.startswith(EXT_PREFIX):
        return EXT_PREFIX + ap
    return ap


def main() -> None:
    if not ZIP.exists():
        print(f"[ERRO] zip nao encontrado: {ZIP}")
        sys.exit(1)

    TARGET.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    extracted = 0

    with zipfile.ZipFile(ZIP) as zf:
        members = zf.infolist()
        total = len(members)
        print(f"A extrair {total} entradas para {TARGET}...", flush=True)
        start = time.time()
        for i, m in enumerate(members, 1):
            dest = TARGET / m.filename
            try:
                if m.is_dir():
                    os.makedirs(long_path(str(dest)), exist_ok=True)
                else:
                    os.makedirs(long_path(str(dest.parent)), exist_ok=True)
                    with zf.open(m) as src, open(long_path(str(dest)), "wb") as out:
                        out.write(src.read())
                    extracted += 1
            except (OSError, FileNotFoundError):
                skipped.append(m.filename)
            if i % 10000 == 0:
                print(f"  {i}/{total} ({i/total*100:.0f}%) extraidos={extracted} skip={len(skipped)} - {time.time()-start:.0f}s", flush=True)

        print(f"Feito: {extracted} ficheiros, {len(skipped)} skip (caminhos longos) em {time.time()-start:.0f}s", flush=True)

    sf110 = TARGET / "SF110-20130704-src"
    projs = [p for p in sf110.iterdir() if p.is_dir() and (p / "build.xml").exists()]
    with_src = [p for p in projs if (p / "src" / "main" / "java").exists()]
    print(f"PROJETOS com build.xml: {len(projs)}", flush=True)
    print(f"PROJETOS com src/main/java: {len(with_src)}", flush=True)

    if skipped:
        skip_log = TARGET.parent / "extraction_skipped.txt"
        skip_log.write_text("\n".join(skipped), encoding="utf-8")
        print(f"Lista de ficheiros saltados: {skip_log}", flush=True)


if __name__ == "__main__":
    main()
