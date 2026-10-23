"""Image I/O helpers that keep Unicode paths working on Windows."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Decode a colour image through NumPy (``cv2.imread`` fails on non-ASCII paths)."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Encode and write an image using the suffix as the format."""
    target = Path(path)
    suffix = target.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {target}")
    encoded.tofile(str(target))
