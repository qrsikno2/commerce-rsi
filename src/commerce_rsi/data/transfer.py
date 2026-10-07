"""Bounded resumable HTTP range downloads; caller verifies the complete digest."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
import urllib.request


def request(url, headers=None):
    return urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0", **(headers or {})})


def fetch_file(url: str, output: Path, size: int, *, direct=False, workers=4,
               chunk_bytes=8 * 1024 * 1024):
    """Strictly check Content-Range; keep completed chunks for interrupted retries."""
    output = Path(output)
    chunks_dir = output.parent / (output.name + ".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    if size < chunk_bytes:
        with opener.open(request(url), timeout=90) as response, output.open("wb") as stream:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("HTML received instead of a data file")
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if output.stat().st_size != size:
            raise ValueError("Incomplete small file transfer")
        return

    def chunk(index):
        start = index * chunk_bytes
        end = min(size, start + chunk_bytes) - 1
        length = end - start + 1
        path = chunks_dir / f"{index:06d}"
        if path.exists() and path.stat().st_size == length:
            return path
        for attempt in range(4):
            try:
                # Range is added to cache key so a broken intermediate cache cannot
                # serve a previous range under the same resolve URL.
                sep = "&" if "?" in url else "?"
                chunk_url = f"{url}{sep}download=true&chunk={index}"
                with opener.open(request(chunk_url, {"Range": f"bytes={start}-{end}"}), timeout=60) as response:
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Server ignored requested range {start}-{end}")
                    with path.open("wb") as stream:
                        remaining = length
                        while remaining:
                            data = response.read(min(1024 * 1024, remaining))
                            if not data:
                                raise ValueError("Truncated range response")
                            stream.write(data)
                            remaining -= len(data)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(min(4, 0.5 * (2 ** attempt)))
        raise AssertionError("unreachable")

    count = (size + chunk_bytes - 1) // chunk_bytes
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(chunk, range(count)))
    with output.open("wb") as stream:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream, length=1024 * 1024)


def clear_chunks(output: Path):
    """Remove only this downloader's numbered scratch pieces after verification."""
    chunks_dir = output.parent / (output.name + ".chunks")
    if chunks_dir.is_dir():
        for child in chunks_dir.iterdir():
            if child.name.isdigit() and child.is_file():
                child.unlink()
        if not any(chunks_dir.iterdir()):
            chunks_dir.rmdir()
