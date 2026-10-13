"""
vendor.py - where the browser libraries come from.

Every library the pages need has
  * CDN URLs      - always work while the phone/desktop has internet, nothing to install
  * a local path  - under static/vendor/, used automatically when the file exists (offline use,
                    phone on your LAN without internet, one less round-trip to a CDN).

`python scripts/fetch_vendor.py` downloads everything into static/vendor/.
app.py asks this module for the import map + URLs and injects them into the HTML pages.

WHY THIS FILE EXISTS (concepts)
  * A browser "import map" is a small JSON table saying "when a script writes  import ... from 'three'  load THIS url".
    By generating the table on the server we can point it at a local copy (fast, offline) or at a CDN
    (nothing to install) without touching the page's JavaScript.
  * A CDN (content delivery network, e.g. jsDelivr) hosts popular libraries; the browser downloads them from there.
  * Phones on your Wi-Fi may have no route to a CDN you can't see - hence the optional local copies.
"""
from pathlib import Path

ROOT = Path(__file__).parent
VENDOR_DIR = ROOT / "static" / "vendor"
VENDOR_URL = "/static/vendor"

THREE_VERSION = "0.160.0"                      # same version the app already used
THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"

# three.js add-ons the pages import (path inside examples/jsm/)
THREE_ADDONS = [
    "loaders/GLTFLoader.js", "loaders/DRACOLoader.js", "controls/OrbitControls.js",
    "utils/BufferGeometryUtils.js", "utils/SkeletonUtils.js",
]
DRACO_FILES = ["draco_decoder.js", "draco_decoder.wasm", "draco_wasm_wrapper.js"]

# name -> list of (relative path under static/vendor, [candidate URLs tried in order])
# "three" is a list of many files (the engine + its add-ons + the Draco mesh decoder); "opencv" and "arjs" are single files.
MANIFEST = {
    "three": (
        [("three/three.module.js", [f"{THREE_CDN}/build/three.module.js"])]
        + [(f"three/addons/{p}", [f"{THREE_CDN}/examples/jsm/{p}"]) for p in THREE_ADDONS]
        + [(f"three/draco/{f}", [f"{THREE_CDN}/examples/jsm/libs/draco/gltf/{f}"]) for f in DRACO_FILES]
    ),
    # OpenCV.js - the official build hosted on docs.opencv.org (single file, WASM embedded, ~10 MB)
    "opencv": [("opencv/opencv.js", [
        "https://docs.opencv.org/4.10.0/opencv.js",
        "https://docs.opencv.org/4.x/opencv.js",
    ])],
    # AR.js (three.js flavour). The project lives at github.com/AR-js-org/AR.js (the original
    # jeromeetienne/AR.js repository was moved there).
    "arjs": [("arjs/ar-threex.js", [
        "https://cdn.jsdelivr.net/gh/AR-js-org/AR.js@3.4.5/three.js/build/ar-threex.js",
        "https://cdn.jsdelivr.net/gh/AR-js-org/AR.js@master/three.js/build/ar-threex.js",
        "https://raw.githack.com/AR-js-org/AR.js/master/three.js/build/ar-threex.js",
    ])],
}

# tesseract.js (https://github.com/naptha/tesseract.js): Tesseract OCR compiled to WebAssembly. Used ONLY as a fallback when the
# server has no Tesseract program. Always loaded from the CDN: besides this script it downloads its own worker, WASM core and
# English language data (a few MB, cached by the browser), so the phone needs internet the first time.
TESSERACTJS_URLS = ["https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js"]

# sanity check used by the fetch script: a download smaller than this is an error page, not the library
MIN_BYTES = {"three/three.module.js": 500_000, "opencv/opencv.js": 3_000_000, "arjs/ar-threex.js": 100_000}


def available(rel: str) -> bool:
    """True when the file exists locally and is not empty (an interrupted download leaves empty files)."""
    p = VENDOR_DIR / rel
    return p.is_file() and p.stat().st_size > 0


def local_url(rel: str) -> str:
    return f"{VENDOR_URL}/{rel}"


def _all_local(group: str) -> bool:
    """All-or-nothing for three.js: mixing a local core with CDN add-ons would load two copies of three.js."""
    return all(available(rel) for rel, _ in MANIFEST[group])


def three_config() -> dict:
    """Import map + Draco decoder location: all-local if the whole three.js set is present, else CDN."""
    if _all_local("three"):
        base = local_url("three")
        return {"importmap": {"imports": {"three": f"{base}/three.module.js", "three/addons/": f"{base}/addons/"}},
                "draco": f"{base}/draco/", "source": "local"}
    return {"importmap": {"imports": {"three": f"{THREE_CDN}/build/three.module.js",
                                      "three/addons/": f"{THREE_CDN}/examples/jsm/"}},
            "draco": f"{THREE_CDN}/examples/jsm/libs/draco/gltf/", "source": "cdn"}


def script_urls(group: str) -> list:
    """URLs to try, in order, for a single-file library ('opencv' or 'arjs'): local copy first."""
    rel, cdn = MANIFEST[group][0]
    return ([local_url(rel)] if available(rel) else []) + list(cdn)


def status() -> dict:
    """What is self-hosted right now (shown in the startup log)."""
    return {"three": three_config()["source"],
            "opencv": "local" if available(MANIFEST["opencv"][0][0]) else "cdn",
            "arjs": "local" if available(MANIFEST["arjs"][0][0]) else "cdn"}
