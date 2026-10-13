Optional self-hosted browser libraries (three.js, OpenCV.js, AR.js).

Run   python scripts/fetch_vendor.py   from the project folder to fill this directory.
Files found here are used instead of the CDN automatically (see vendor.py); anything missing
falls back to the CDN, so an empty folder is fine.
