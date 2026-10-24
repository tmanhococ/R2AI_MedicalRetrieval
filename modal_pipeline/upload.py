"""Resumable upload of the prepared bundle to the Modal volume (vb_common.VOLUME_NAME) under /data.

Files already on the volume with the same size are skipped, so the script can be
re-run after a dropped connection.

    python modal_pipeline/upload.py artifacts/modal_upload_v2
    python modal_pipeline/upload.py artifacts/modal_upload_v2 --only query.parquet official_corpus.parquet chunks/part_00000.parquet
"""
from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
import sys
import time

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vb_common import VOLUME_NAME  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", type=Path, default=Path("artifacts/modal_upload_v2"))
    ap.add_argument("--only", nargs="+", default=None, help="glob patterns relative to src (posix paths)")
    args = ap.parse_args()
    src = args.src
    if not (src / "chunks").is_dir():
        sys.exit(f"{src} has no chunks/ folder; run prepare_upload.py first")
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    try:
        remote = {e.path.lstrip("/"): e.size for e in vol.listdir("/data", recursive=True)}
    except Exception:  # nothing uploaded yet
        remote = {}

    files = sorted(p for p in src.rglob("*") if p.is_file())
    if args.only:
        files = [p for p in files if any(fnmatch(p.relative_to(src).as_posix(), g) for g in args.only)]
        if not files:
            sys.exit(f"--only matched no files under {src}")
    total = sum(p.stat().st_size for p in files)
    sent = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        rel = "data/" + p.relative_to(src).as_posix()
        size = p.stat().st_size
        if remote.get(rel) == size:
            sent += size
            continue
        for attempt in range(1, 6):
            try:
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(str(p), "/" + rel)
                break
            except Exception as exc:  # flaky home connections
                print(f"  retry {attempt}/5 for {rel}: {exc}", flush=True)
                time.sleep(10 * attempt)
        else:
            sys.exit(f"giving up on {rel}; re-run the script to resume")
        sent += size
        mbps = sent / 1e6 / max(1e-6, time.time() - t0)
        print(f"[{i}/{len(files)}] {rel} ({size / 1e6:.0f} MB) - {sent / 1e9:.2f}/{total / 1e9:.2f} GB, {mbps:.1f} MB/s",
              flush=True)
    print("Upload complete.")


if __name__ == "__main__":
    main()
