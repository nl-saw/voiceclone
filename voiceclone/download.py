"""Small streaming file downloader (progress bar, .part + atomic rename).

Used instead of coqui's ModelManager because the scarf.sh gateway is not
reachable from every network; HuggingFace resolve URLs work directly.
"""

from __future__ import annotations

import time
from pathlib import Path


def fetch(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download ``url`` to ``dest`` (skips if already present and non-empty)."""
    import requests

    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        last_print = [0.0]
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if total and now - last_print[0] > 0.5:
                    last_print[0] = now
                    print(
                        f"\r    {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                        f"({100 * done // total}%)",
                        end="",
                        flush=True,
                    )
    if total:
        print()
    tmp.rename(dest)
    return dest
