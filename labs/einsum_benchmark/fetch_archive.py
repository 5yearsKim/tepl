#!/usr/bin/env python3
"""Download the complete pinned Zenodo archive with bounded parallel ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import urllib.request

URL = 'https://zenodo.org/records/11477304/files/instances.zip?download=1'

SIZE = 551789940
MD5 = 'b476f37d39a5b8a978e8230375326eb3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be between 1 and 8')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.download')
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o644)
    os.ftruncate(fd, SIZE)
    chunk = 8 * 1024 * 1024

    def fetch(start):
        end = min(start + chunk, SIZE) - 1
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('incorrect archive byte range')
                    offset = start
                    while data := response.read(256 * 1024):
                        if offset + len(data) > end + 1:
                            raise ValueError('oversized archive byte range')
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], offset + written)
                        offset += len(data)
                    if offset != end + 1:
                        raise ValueError('truncated archive byte range')
                return end - start + 1
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(fetch, start) for start in range(0, SIZE, chunk)]
            done = 0
            for future in as_completed(futures):
                done += future.result()
                print(f'{done}/{SIZE} bytes ({done / SIZE:.1%})', flush=True)
    finally:
        os.close(fd)
    check = hashlib.md5()
    with temporary.open('rb') as f:
        while data := f.read(1024 * 1024):
            check.update(data)
    if check.hexdigest() != MD5:
        raise ValueError('official archive MD5 mismatch')
    temporary.replace(args.out)
    print(f'Verified {MD5}: {args.out}')


if __name__ == '__main__':
    main()
