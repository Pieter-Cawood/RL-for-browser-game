"""Percent-based cropping utilities for canvas frames."""
from __future__ import annotations

import numpy as np
from .config import CropPercent
from .utils import clamp

class PercentCropper:
    """Crop an image using percent-based band definition."""
    def __init__(self, cfg: CropPercent):
        self.cfg = cfg

    def crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        bo = clamp(self.cfg.bottom_offset_percent, 0.0, 100.0)
        hp = clamp(self.cfg.height_percent, 0.0, 100.0)
        lp = clamp(self.cfg.left_percent, 0.0, 100.0)
        rp = clamp(self.cfg.right_percent, 0.0, 100.0)

        band_h = int(round(h * (hp / 100.0)))
        bottom_offset = int(round(h * (bo / 100.0)))

        y2 = max(0, h - bottom_offset)
        y1 = max(0, y2 - band_h)
        x1 = int(round(w * (lp / 100.0)))
        x2 = int(round(w * (1.0 - rp / 100.0)))

        # Clamp + validate
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        if y2 <= y1 or x2 <= x1:
            return img
        return img[y1:y2, x1:x2]
