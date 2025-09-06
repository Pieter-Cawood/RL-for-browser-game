"""Configuration loading from .env and environment variables for the browser/canvas layer."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from .utils import parse_bool

ButtonType = Literal["css", "role", "text", "xpath"]

class ButtonSpec(TypedDict):
    type: ButtonType
    value: str

@dataclass(frozen=True)
class CropPercent:
    bottom_offset_percent: float = 0.0  # shift band up from bottom (0..100)
    height_percent: float = 30.0        # band height (0..100)
    left_percent: float = 0.0           # crop from left (0..100)
    right_percent: float = 0.0          # crop from right (0..100)

@dataclass(frozen=True)
class ClickBehavior:
    retries: int = 6
    backoff_ms: int = 300
    stability_ms: int = 200
    wait_visible_ms: int = 10_000

@dataclass(frozen=True)
class BrowserSettings:
    headless: bool = False
    slow_mo_ms: int = 0

@dataclass(frozen=True)
class BrowserGameConfig:
    url: str
    canvas_selector: str
    reaction_movement_time_ms: float
    intro_buttons: List[ButtonSpec]
    replay_buttons: List[ButtonSpec]
    crop: CropPercent
    click: ClickBehavior
    browser: BrowserSettings

    @staticmethod
    def from_env() -> "BrowserGameConfig":
        load_dotenv()

        url = os.getenv("GAME_URL", "")
        canvas_selector = os.getenv("CANVAS_SELECTOR", "#game-area canvas")
        reaction_movement_time_ms = float(os.getenv("REACTION_MOVEMENT_TIME_MS", "50"))

        try:
            intro_buttons = json.loads(os.getenv("INTRO_BUTTONS", "[]"))
            replay_buttons = json.loads(os.getenv("REPLAY_BUTTONS", "[]"))
            assert isinstance(intro_buttons, list), isinstance(replay_buttons, list)
        except Exception:
            intro_buttons = []
            replay_buttons = []
            

        crop = CropPercent(
            bottom_offset_percent=float(os.getenv("CROP_BOTTOM_OFFSET_PERCENT", "0")),
            height_percent=float(os.getenv("CROP_HEIGHT_PERCENT", "30")),
            left_percent=float(os.getenv("CROP_LEFT_PERCENT", "0")),
            right_percent=float(os.getenv("CROP_RIGHT_PERCENT", "0")),
        )

        click = ClickBehavior(
            retries=int(os.getenv("CLICK_RETRIES", "6")),
            backoff_ms=int(os.getenv("CLICK_BACKOFF_MS", "300")),
            stability_ms=int(os.getenv("BUTTON_STABILITY_MS", "200")),
            wait_visible_ms=int(os.getenv("BUTTON_WAIT_VISIBLE_MS", "10000")),
        )

        browser = BrowserSettings(
            headless=parse_bool(os.getenv("HEADLESS"), False),
            slow_mo_ms=int(os.getenv("SLOW_MO_MS", "0")),
        )

        return BrowserGameConfig(
            url=url,
            canvas_selector=canvas_selector,
            reaction_movement_time_ms=reaction_movement_time_ms,
            intro_buttons=intro_buttons,
            replay_buttons=replay_buttons,
            crop=crop,
            click=click,
            browser=browser,
        )

