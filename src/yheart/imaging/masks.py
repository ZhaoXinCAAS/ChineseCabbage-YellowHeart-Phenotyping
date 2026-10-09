"""Read lossless region masks referenced by annotation JSON."""

from pathlib import Path

import cv2
import numpy as np


def read_mask(path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    """Load a binary PNG without resizing or silently changing coordinates."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable mask: {path}")
    if image.shape != shape[:2]:
        raise ValueError(f"Mask shape mismatch: {path}: {image.shape} vs {shape[:2]}")
    if not np.isin(image, [0, 255]).all():
        raise ValueError(f"Expected binary 0/255 mask: {path}")
    return image > 0
