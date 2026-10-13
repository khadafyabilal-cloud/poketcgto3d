#!/usr/bin/env python3
"""
Download the browser libraries (three.js + add-ons, OpenCV.js, AR.js) into static/vendor/ so the
app can serve them itself. Optional: without it the pages load the same files from their CDNs.

    python scripts/fetch_vendor.py            # everything missing
    python scripts/fetch_vendor.py --force    # re-download
    python scripts/fetch_vendor.py --only opencv arjs

Self-hosting is worth it when the phone is on your LAN without good internet, or to avoid the
~10 MB OpenCV.js download from docs.opencv.org on every new device (the browser caches it after).
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests
import vendor


def fetch(rel, urls, force):
    dest = vendor.VENDOR_DIR / rel
    if dest.exists() and not force:
        return "have", None
    last = "no URL"
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.content
            need = vendor.MIN_BYTES.get(rel, 1)
            if len(data) < need:
                last = f"{url}: only {len(data)} bytes (expected >= {need})"; continue
            if rel.endswith(".wasm") and data[:4] != b"\0asm":
                last = f"{url}: not a WebAssembly file"; continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data); os.replace(tmp, dest)            # atomic: no half-written files
            return "got", f"{len(data) // 1024} KB from {url}"
        except Exception as e:
            last = f"{url}: {type(e).__name__}: {e}"
    return "FAILED", last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    ap.add_argument("--only", nargs="+", choices=list(vendor.MANIFEST), help="only these groups")
    args = ap.parse_args()

    failed = 0
    for group, entries in vendor.MANIFEST.items():
        if args.only and group not in args.only:
            continue
        print(f"[{group}]")
        for rel, urls in entries:
            state, detail = fetch(rel, urls, args.force)
            failed += state == "FAILED"
            print(f"  {state:6} {rel}" + (f"  ({detail})" if detail else ""))
    print("\nself-hosted now:", vendor.status())
    if failed:
        print(f"{failed} file(s) failed - the app still works: it falls back to the CDN for those.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
