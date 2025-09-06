"""Canvas navigation and frame capture using Playwright."""
from __future__ import annotations

from typing import Generator, Optional, Iterable

import cv2
import numpy as np
from playwright.sync_api import sync_playwright, Page

from .config import BrowserGameConfig, ButtonSpec
from .clicker import ButtonClicker
from .crop import PercentCropper

class CanvasRunner:
    """Open the game, click through intro steps, and stream canvas frames."""
    def __init__(self, cfg: BrowserGameConfig):
        self.cfg = cfg

    def _launch(self):
        return sync_playwright()

    def _click_intro(self, page: Page, buttons: Iterable[ButtonSpec]) -> None:
        clicker = ButtonClicker(page, self.cfg.click)
        for spec in buttons:
            ok = clicker.click(spec)
            if not ok:
                print(f"Continuing despite failure: {spec}")

    def stream(self, cropper: Optional[PercentCropper] = None) -> Generator[np.ndarray, None, None]:
        with self._launch() as p:
            browser = p.chromium.launch(headless=self.cfg.browser.headless, slow_mo=self.cfg.browser.slow_mo_ms)
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()

            page.goto(self.cfg.url, wait_until="domcontentloaded")
            self._click_intro(page, self.cfg.intro_buttons)

            canvas = page.wait_for_selector(self.cfg.canvas_selector, timeout=25_000, state="visible")
            print("Game canvas ready. Press ESC to quit.")

            while True:
                box = canvas.bounding_box()
                if not box:
                    continue
                clip = {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
                png = page.screenshot(clip=clip)
                frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
                if cropper:
                    frame = cropper.crop(frame)
                yield frame
