"""Download the Helsinki neonatal EEG dataset (Zenodo record 2547147) with retries + checksums."""
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REC = "2547147"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

with urllib.request.urlopen(f"https://zenodo.org/api/records/{REC}", timeout=60) as r:
    meta = json.load(r)

files = [(f["key"], f["size"], f["checksum"], f["links"]["self"]) for f in meta["files"]]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(key, size, checksum, url, tries=6):
    dest = os.path.join(OUT, key)
    expected = checksum.split(":")[-1].lower()
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        if md5(dest) == expected:
            return (key, "cached", size)
    for attempt in range(1, tries + 1):
        try:
            tmp = dest + ".part"
            req = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            got = md5(tmp)
            if got != expected:
                raise IOError(f"checksum mismatch {got} != {expected}")
            os.replace(tmp, dest)
            return (key, "ok", size)
        except Exception as e:  # noqa: BLE001
            if attempt == tries:
                return (key, f"FAILED: {e}", size)
            time.sleep(3 * attempt)
    return (key, "FAILED retries", size)


done, total_bytes, fails = 0, 0, []
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(fetch, *f): f[0] for f in files}
    for fut in as_completed(futs):
        key, status, size = fut.result()
        done += 1
        total_bytes += size if status in ("ok", "cached") else 0
        if status.startswith("FAILED"):
            fails.append(key)
        if done % 10 == 0 or status.startswith("FAILED"):
            print(f"[{done}/{len(files)}] {key}: {status} ({total_bytes/1e9:.2f} GB)", flush=True)

print(f"COMPLETE: {done - len(fails)}/{len(files)} files, {total_bytes/1e9:.2f} GB", flush=True)
if fails:
    print("FAILED FILES:", " ".join(fails), flush=True)
    sys.exit(1)
