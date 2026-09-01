"""GPU memory helpers shared by the synthesis engine and the training guard."""

from __future__ import annotations


def free_vram_gib() -> float | None:
    """Free VRAM in GiB on the primary CUDA device, or None if not determinable.

    Reports what the driver can still allocate — i.e. it *does* account for
    memory held by other processes (e.g. a loaded LLM), which is exactly the
    case these guards exist to catch.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:  # noqa: BLE001 — a query failure must never block the caller
        return None
