"""
Browser automation utilities:
- BrowserGameConfig: load .env config
- ButtonClicker: robust intro button clicking
- CanvasRunner: open page, click intros, stream canvas frames
- PercentCropper: percent-based cropping (bottom/height/left/right)
"""

from .config import BrowserGameConfig, CropPercent, ClickBehavior, BrowserSettings
from .clicker import ButtonClicker
from .canvas import CanvasRunner
from .crop import PercentCropper

__all__ = [
    "BrowserGameConfig", "CropPercent", "ClickBehavior", "BrowserSettings",
    "ButtonClicker", "CanvasRunner", "PercentCropper",
]
