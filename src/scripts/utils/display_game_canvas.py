"""
Show the cropped Red Bull game canvas in a window.

Run from the project root:
    python -m src.scripts.utils.display_game_canvas

Prereqs:
    pip install -r requirements.txt
    python -m playwright install chromium
"""

import cv2
from src.app.browser_automation import BrowserGameConfig, CanvasRunner, PercentCropper


def main() -> None:
    # Load configuration from .env
    cfg = BrowserGameConfig.from_env()

    # Compose runner + cropper from the browser_automation layer
    runner = CanvasRunner(cfg)
    cropper = PercentCropper(cfg.crop)

    # Stream frames from the canvas
    for frame in runner.stream(cropper=cropper):
        cv2.imshow("Game Canvas", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
