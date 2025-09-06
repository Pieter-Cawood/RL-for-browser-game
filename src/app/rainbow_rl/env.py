# -*- coding: utf-8 -*-
"""Rainbow-compatible environment that drives a browser canvas via Playwright.

Key improvements vs. the original:
- Fixed bugs (duplicate action_space, incorrect last_score init, None image guards).
- Robust Playwright lifecycle with bounded retries and safe teardown.
- Logging instead of prints, plus rich docstrings and type hints.
- Safer OpenCV UI (guarded for headless use), consistent window reuse.
- Clear feature-vector vs. image observation paths with history buffer.
- Utility functions documented and edge cases handled (empty frames, ROI bounds).

Usage (sketch):
    env = BrowserGameEnv(cfg, device=torch.device("cpu"))
    state = env.reset()
    for _ in range(1000):
        state, reward, done = env.step(action)
        if done:
            break
    env.close()
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import logging
import math
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image, ImageSequence
from playwright.sync_api import Page, Locator, TimeoutError as PWTimeout, sync_playwright

from src.app.browser_automation.clicker import ButtonClicker
from src.app.browser_automation.config import BrowserGameConfig, CropPercent
from src.app.browser_automation.crop import PercentCropper

import base64

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Default console handler (library-friendly: INFO by default; let apps override)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Constants
PREVIEW_SIZE = 84  # DQN/Rainbow standard input side length
HUD_HEIGHT = 28
DEFAULT_MAX_STEPS = 100000


# --------------------------------------------------------------------------- #
# Dot detection & synthetic rendering
# --------------------------------------------------------------------------- #

def detect_dot_x_norm(
    frame_bgr: np.ndarray,
    gauge_y_frac: Tuple[float, float] = (0.05, 0.25),
    min_blob_area_frac: float = 0.0002,
    v_thresh: int = 70,
) -> Tuple[bool, float]:
    """Detect the dark gauge dot and return (found, x_norm) with x_norm ∈ [0, 1].

    Args:
        frame_bgr: BGR image from canvas.
        gauge_y_frac: Fractional vertical ROI (start, end) to limit search.
        min_blob_area_frac: Area threshold (fraction of ROI) to reject noise.
        v_thresh: Threshold on HSV V channel (binary inv) for dark-on-bright.

    Returns:
        (found, x_norm)
    """
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return False, 0.5

    y0 = int(max(0, min(h - 1, h * gauge_y_frac[0])))
    y1 = int(max(1, min(h,     h * gauge_y_frac[1])))
    if y1 <= y0:
        y0, y1 = 0, h

    roi = frame_bgr[y0:y1, :]
    V = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    dot_mask = cv2.threshold(V, v_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones((3, 3), np.uint8)
    dot_mask = cv2.morphologyEx(dot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dot_mask = cv2.morphologyEx(dot_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(dot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    min_area = max(1, int(min_blob_area_frac * roi_area))

    cx = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cxi = int(M["m10"] / M["m00"])  # x within ROI
        if area > best_area:
            best_area = area
            cx = cxi

    if cx is None:
        return False, 0.5

    x_norm = float(np.clip(cx / max(1, (w - 1)), 0.0, 1.0))
    return True, x_norm


def render_dot_gray_84x84(x_norm: float, radius: int = 3, draw_center: bool = True) -> torch.Tensor:
    """Render a white 84×84 image with a black dot at x_norm and optional center tick.

    Returns:
        torch.Tensor shape [84, 84] in [0, 1].
    """
    img = np.full((PREVIEW_SIZE, PREVIEW_SIZE), 255, dtype=np.uint8)
    if draw_center:
        cx = PREVIEW_SIZE // 2
        cv2.line(img, (cx, 0), (cx, PREVIEW_SIZE - 1), 220, 1)
    x = int(round(np.clip(x_norm, 0.0, 1.0) * (PREVIEW_SIZE - 1)))
    y = PREVIEW_SIZE // 2
    cv2.circle(img, (x, y), radius, 0, -1, lineType=cv2.LINE_AA)
    return torch.tensor(img, dtype=torch.float32).div_(255.0)


def default_done_fn(env: "BrowserGameEnv", frame: np.ndarray, steps_in_episode: int, max_steps: int = DEFAULT_MAX_STEPS) -> bool:
    """Default termination: episode ends if max_steps exceeded or a leaderboard score appears."""
    if steps_in_episode >= max_steps:
        logger.info("Episode max steps reached: %d", steps_in_episode)
        return True
    score = env._read_leaderboard_score()
    return score is not None


 # -------------------------------- Rewards ------------------------------ #

def center_reward_triangle(x_norm: float, *_ignore, **__ignore) -> float:
    """Linear triangular reward in [-1, +1]: +1 at x=0.5, -1 at edges."""
    x = float(np.clip(x_norm, 0.0, 1.0))
    r = 1.0 - 4.0 * abs(x - 0.5)  # 0.5 -> +1, 0/1 -> -1
    return float(np.clip(r, -1.0, 1.0))

def center_reward_exp_triangle(x_norm: float, k: float = 6.0, *_ignore, **__ignore) -> float:
    """Exponential 'triangle' reward in [-1, +1], sharper near center as k increases."""
    x = float(np.clip(x_norm, 0.0, 1.0))
    d = abs(x - 0.5) / 0.5  # 0 at center, 1 at edges
    if k <= 0:
        r = 1.0 - 2.0 * d
    else:
        num = -math.expm1(-k * d)   # 1 - exp(-k d)
        den = -math.expm1(-k)       # 1 - exp(-k)
        g = num / den if den != 0.0 else d
        r = 1.0 - 2.0 * g
    return float(np.clip(r, -1.0, 1.0))

def center_reward(x_norm: float, tol: float = 0.01, alpha: float = 12.0, beta: float = 3.0) -> float:
    """Exponential reward in [0, 1]:
    - 1.0 inside a tight center band (±tol).
    - Rapid decay toward 0 outside that band.
    """
    x = float(np.clip(x_norm, 0.0, 1.0))
    d = abs(x - 0.5) / 0.5
    if d <= tol:
        return 1.0
    d_out = (d - tol) / (1.0 - tol)
    return float(np.exp(-alpha * (d_out ** beta)))


# --------------------------------------------------------------------------- #
# Browser-backed Env
# --------------------------------------------------------------------------- #

class BrowserGameEnv:
    """Rainbow-style environment over a web canvas (Playwright + OpenCV).

    API:
        reset() -> torch.Tensor [H, ...] float in [0,1]
        step(a: int) -> (state, reward, done)
        action_space() -> int
        train() / eval()
        close()

    Actions:
        0 = no-op
        1..3 = ArrowLeft with 1..3 taps
        4..6 = ArrowRight with 1..3 taps
    """

    # ------------------------------ Construction --------------------------- #

    def __init__(
        self,
        cfg: BrowserGameConfig,
        device: torch.device,
        done_fn=default_done_fn,
        use_vector_obs: bool = True,
        crop: Optional[CropPercent] = None,
        history_length: int = 3,
        synthetic_input: bool = True,    # render dot on white 84x84 (image obs only)
        dark_foreground: bool = True,    # retained for compatibility; currently unused
        feature_keys: Tuple[str, ...] = ("x", "dx_before", "dx_after"),
        max_session_retries: int = 3,
        display_input_window: bool = True,  # Displays input going to the model in a window
        display_agent_playing: bool = True, # Displays agent animation playing the game
    ):
        self.cfg = cfg
        self.device = device
        self.history_length = int(history_length)
        self.done_fn = done_fn
        self.synthetic_input = bool(synthetic_input)
        self.dark_foreground = bool(dark_foreground)  # placeholder for future use
        self.display_input_window = bool(display_input_window)
        self.display_agent_playing_window = bool(display_agent_playing)
        self.training = True
        self.state_buffer: deque[torch.Tensor] = deque([], maxlen=self.history_length)
        self.steps_in_episode = 0
        self.last_score: Optional[int] = None

        self.cropper = PercentCropper(crop or cfg.crop)

        # Playwright/session handles
        self._pw = None
        self._browser = None
        self._context = None
        self._page: Optional[Page] = None
        self._canvas: Optional[Locator] = None
        self._initialized = False

        # UI windows flags
        self._gray_window_inited = False
        self._combined_window_initialized = False

        # Observation config
        self.use_vector_obs = bool(use_vector_obs)
        self.feature_keys = tuple(feature_keys)
        self.feature_dim = len(self.feature_keys)

        # Motion tracking
        self._x_prev: float = 0.5
        self._dx_prev: float = 0.0

        # Bounds and timing
        self.lower_limit_x: float = 0.1
        self.upper_limit_x: float = 0.9
        self.max_taps: int = 3
        self.key_hold_ms: int = 5
        self.step_wait_ms: int = 5
        self.inter_tap_ms: int = 5
        self.max_session_retries: int = max(1, int(max_session_retries))

        # Actions mapping (NOOP, Left×N, Right×N)
        self.actions = {
            0: None,
            1: "x1ArrowLeft",
            2: "x2ArrowLeft",
            3: "x3ArrowLeft",
            4: "x1ArrowRight",
            5: "x2ArrowRight",
            6: "x3ArrowRight",
        }

        # Media (agent overlays)
        base_dir = os.path.dirname(__file__)
        media_dir = os.path.join(base_dir, "media")

        def load_and_resize(img_path: str, scale: float = 0.2) -> Optional[np.ndarray]:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Could not load image: %s", img_path)
                return None
            h, w = img.shape[:2]
            resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            return resized

        def load_gif_frames(path: str, scale: float = 0.5) -> list[np.ndarray]:
            frames: list[np.ndarray] = []
            try:
                pil_img = Image.open(path)
            except Exception as e:
                logger.warning("Could not load GIF '%s': %s", path, e)
                return frames
            for frame in ImageSequence.Iterator(pil_img):
                rgb = frame.convert("RGB")
                w, h = rgb.size
                rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=Image.Resampling.LANCZOS)
                bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                frames.append(bgr)
            return frames

        self.agent_imgs: dict[int, Optional[np.ndarray]] = {
            0: load_and_resize(os.path.join(media_dir, "agent-action-none.png")),
            1: load_and_resize(os.path.join(media_dir, "agent-action-left.png")),
            2: load_and_resize(os.path.join(media_dir, "agent-action-left.png")),
            3: load_and_resize(os.path.join(media_dir, "agent-action-left.png")),
            4: load_and_resize(os.path.join(media_dir, "agent-action-right.png")),
            5: load_and_resize(os.path.join(media_dir, "agent-action-right.png")),
            6: load_and_resize(os.path.join(media_dir, "agent-action-right.png")),
        }

        self.binary_agent_frames: list[np.ndarray] = load_gif_frames(
            os.path.join(media_dir, "agent-binary-code.gif")
        )
        self._binary_agent_frame_index: int = 0

    # ------------------------------ Public API ----------------------------- #

    def action_space(self) -> int:
        """Number of discrete actions."""
        return len(self.actions)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def close(self) -> None:
        """Destroy windows and tear down Playwright."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._teardown_session()

    # ---------------------------- Session / Setup -------------------------- #

    def _start_session(self) -> None:
        """Start a fresh Playwright session, open page, click intros, wait for canvas."""
        logger.info("Starting Playwright session...")
        self._pw = sync_playwright().start()

        launch_args = [
            "--disable-features=BlockThirdPartyCookies,SameSiteByDefaultCookies,CookiesWithoutSameSiteMustBeSecure",
            "--allow-third-party-cookies",
            "--disable-site-isolation-trials",
        ]

        # Try to launch with Chrome channel; fall back to default Chromium if unavailable
        try:
            self._browser = self._pw.chromium.launch(
                headless=self.cfg.browser.headless,
                slow_mo=self.cfg.browser.slow_mo_ms,
                args=launch_args,
                channel="chrome",
            )
        except Exception as e:
            logger.warning("Could not launch with channel='chrome' (%s). Falling back to default.", e)
            self._browser = self._pw.chromium.launch(
                headless=self.cfg.browser.headless,
                slow_mo=self.cfg.browser.slow_mo_ms,
                args=launch_args,
            )

        self._context = self._browser.new_context(
            bypass_csp=True,
            ignore_https_errors=True,
            java_script_enabled=True,
            viewport={"width": 1280, "height": 900},
            user_agent=None,
        )
        self._page = self._context.new_page()
        self._page.goto(self.cfg.url, wait_until="domcontentloaded")

        # Dismiss intro modals/buttons
        clicker = ButtonClicker(self._page, self.cfg.click)
        for spec in self.cfg.intro_buttons:
            try:
                clicker.click(spec)
            except Exception:
                # Non-fatal; the canvas wait is the real gate.
                logger.debug("Intro click failed for spec: %s", spec)

        # Wait for the canvas
        try:
            self._canvas = self._page.wait_for_selector(
                self.cfg.canvas_selector,
                timeout=25_000,
                state="visible",
            )
        except PWTimeout as e:
            logger.error("Canvas selector did not appear: %s", e)
            raise

        self._initialized = True
        logger.info("Playwright session initialized.")

    def _reinit_session(self) -> None:
        """Hard reset the whole browser/game with bounded retries."""
        logger.info("Reinitializing Playwright session...")
        for attempt in range(self.max_session_retries):
            try:
                self._teardown_session()
                self._start_session()
                return
            except Exception as e:
                logger.warning("Reinit attempt %d/%d failed: %s", attempt + 1, self.max_session_retries, e)
        # Final attempt raises to caller
        self._teardown_session()
        self._start_session()

    def _teardown_session(self) -> None:
        """Close everything safely."""
        logger.info("Tearing down Playwright session...")
        for attr in ("_context", "_browser", "_pw"):
            try:
                obj = getattr(self, attr)
                if obj:
                    # The API names differ: context/browser have .close(), playwright has .stop()
                    if attr == "_pw":
                        obj.stop()
                    else:
                        obj.close()
            except Exception:
                logger.debug("Ignoring error tearing down %s", attr)
            finally:
                setattr(self, attr, None)

        self._page = None
        self._canvas = None
        self._initialized = False
        self._combined_window_initialized = False
        self._gray_window_inited = False

    def _init_once(self) -> None:
        if not self._initialized:
            self._start_session()

    # ---------------------------- Image helpers ---------------------------- #

    @staticmethod
    def _to_84x84_gray(frame_bgr: np.ndarray) -> torch.Tensor:
        """HSV-V channel + CLAHE + letterbox resize to 84×84 (uint8->float[0,1])."""
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            small = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE), dtype=np.uint8)
            return torch.tensor(small, dtype=torch.float32).div_(255.0)
        v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        scale = min(PREVIEW_SIZE / w, PREVIEW_SIZE / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        nn = cv2.resize(v, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE), dtype=np.uint8)
        y0 = (PREVIEW_SIZE - nh) // 2
        x0 = (PREVIEW_SIZE - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = nn
        return torch.tensor(canvas, dtype=torch.float32).div_(255.0)

    @staticmethod
    def _to_84x84_bgr(frame_bgr: np.ndarray) -> np.ndarray:
        """Aspect-preserving letterbox resize to 84×84 (BGR)."""
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)
        scale = min(PREVIEW_SIZE / w, PREVIEW_SIZE / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        nn = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)
        y0 = (PREVIEW_SIZE - nh) // 2
        x0 = (PREVIEW_SIZE - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = nn
        return canvas

    # ------------------------------ UI helpers ----------------------------- #

    def _show_combo(
        self,
        color84_bgr: np.ndarray,
        gray84: torch.Tensor,
        reward: Optional[float] = None,
        win_name: str = "Canvas+Gray+Reward",
    ) -> None:
        """Compose a single window with color (top), gray (mid), and reward HUD (bottom)."""
        try:
            gray_img = (gray84.detach().cpu().numpy() * 255).astype(np.uint8)
            if gray_img.ndim == 2:
                gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            else:
                gray_bgr = gray_img

            H, W = PREVIEW_SIZE, PREVIEW_SIZE
            canvas = np.zeros((H + H + HUD_HEIGHT, W, 3), dtype=np.uint8)

            # Top: color, Mid: gray
            color_84 = (
                color84_bgr
                if (color84_bgr.shape[0] == H and color84_bgr.shape[1] == W)
                else self._to_84x84_bgr(color84_bgr)
            )
            canvas[0:H, :, :] = color_84
            canvas[H:2 * H, :, :] = gray_bgr

            # Bottom: reward HUD
            if reward is not None:
                r = float(np.clip(reward, -1.0, 1.0))
                t = (r + 1.0) / 2.0  # [-1,1] -> [0,1]
                y0 = 2 * H
                y1 = 2 * H + HUD_HEIGHT - 1
                bar_x0, bar_x1 = 4, W - 4
                bar_y = (y0 + y1) // 2

                # Bar background & outline
                cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (60, 60, 60), 10)
                cv2.rectangle(canvas, (bar_x0, bar_y - 10), (bar_x1, bar_y + 10), (100, 100, 100), 1)

                # Filled portion
                x_fill = int(bar_x0 + t * (bar_x1 - bar_x0))
                cv2.line(canvas, (bar_x0, bar_y), (x_fill, bar_y), (0, 200, 0), 10)

                # Center tick
                cx = (bar_x0 + bar_x1) // 2
                cv2.line(canvas, (cx, bar_y - 10), (cx, bar_y + 10), (255, 255, 255), 1)

                # Text
                txt = f"reward: {r:+.3f} (-1..+1)"
                cv2.putText(canvas, txt, (6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

            if not self._gray_window_inited:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, PREVIEW_SIZE * 4, (PREVIEW_SIZE * 2 + HUD_HEIGHT) * 4)
                self._gray_window_inited = True

            cv2.imshow(win_name, canvas)
            cv2.waitKey(1)
        except Exception as e:
            # In headless environments, OpenCV windows will fail; mute errors.
            logger.debug("OpenCV preview skipped: %s", e)

    def _show_combined_agent_view(self, action: int) -> None:
        """Show a side-by-side animation (binary code) and the current action image."""
        try:
            action_img = self.agent_imgs.get(action)
            if action_img is None:
                return
            if not self.binary_agent_frames:
                return

            binary_frame = self.binary_agent_frames[self._binary_agent_frame_index]
            self._binary_agent_frame_index = (self._binary_agent_frame_index + 1) % len(self.binary_agent_frames)

            # Match heights
            h = min(action_img.shape[0], binary_frame.shape[0])
            if h <= 0:
                return
            action_img_resized = cv2.resize(
                action_img, (int(action_img.shape[1] * h / action_img.shape[0]), h)
            )
            binary_frame_resized = cv2.resize(
                binary_frame, (int(binary_frame.shape[1] * h / binary_frame.shape[0]), h)
            )

            combined = cv2.hconcat([binary_frame_resized, action_img_resized])
            win_name = "Redbull Stalen Ros RL Agent"
            if not self._combined_window_initialized:
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                self._combined_window_initialized = True

            cv2.imshow(win_name, combined)
            cv2.waitKey(1)
        except Exception as e:
            logger.debug("Agent combined view skipped: %s", e)

    # ---------------------- Canvas capture & keyboard ---------------------- #

    def _grab_canvas_bgr(self) -> Tuple[np.ndarray, bool]:
        if not (self._page and self._canvas):
            return np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8), False
        try:
            # Wait for the next paint, then read the canvas
            data_url = self._page.evaluate(
                """sel => new Promise(r => requestAnimationFrame(() => {
                    const el = document.querySelector(sel);
                    const canvas = el instanceof HTMLCanvasElement ? el : el.querySelector('canvas');
                    r(canvas.toDataURL('image/png'));
                }))""",
                self.cfg.canvas_selector,
            )
            png = base64.b64decode(data_url.split(",")[1])
            rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)  # H×W×4
            if rgba is None:
                return np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8), False

            # Alpha-composite onto a stable background (black here)
            if rgba.shape[2] == 4:
                bgr, a = rgba[:, :, :3], rgba[:, :, 3:4] / 255.0
                bg = np.zeros_like(bgr)
                bgr = (bgr * a + bg * (1 - a)).astype(np.uint8)
            else:
                bgr = rgba

            return self.cropper.crop(bgr), True
        except Exception as e:
            logger.debug("Canvas grab failed: %s", e)
            return np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8), False

    def _do_action(self, a: int) -> None:
        """Simulate the key action by tapping ArrowLeft/ArrowRight up to max_taps."""
        if not self._page:
            return
        spec = self.actions.get(a)
        if not spec:
            return

        # Parse "x3ArrowLeft" -> taps=3, key="ArrowLeft"
        taps, key = 1, str(spec).strip()
        m = re.match(r"^x(\d+)(.+)$", key)
        if m:
            taps = max(1, int(m.group(1)))
            key = m.group(2)

        taps = min(int(taps), int(self.max_taps))

        try:
            for i in range(taps):
                self._page.keyboard.down(key)
                self._page.wait_for_timeout(self.key_hold_ms)
                self._page.keyboard.up(key)
                if i < taps - 1:
                    self._page.wait_for_timeout(self.inter_tap_ms)
        except Exception as e:
            logger.debug("Action send failed for '%s': %s", spec, e)

    # ---------------------------- Feature builder -------------------------- #

    def _make_features(self, x_after: float, dx_before: float, dx_after: float) -> torch.Tensor:
        """Assemble the feature vector based on configured feature_keys."""
        feats: list[float] = []
        for k in self.feature_keys:
            if k == "x":
                feats.append(float(np.clip(x_after, 0.0, 1.0)))
            elif k == "dx_before":
                feats.append(float(dx_before))
            elif k == "dx_after":
                feats.append(float(dx_after))
            elif k == "dist_center":
                feats.append(float((x_after - 0.5) / 0.5))
            else:
                raise ValueError(f"Unknown feature key: {k}")
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    # ------------------------------- Reset --------------------------------- #

    def _reset_buffer(self) -> None:
        """Clear the state buffer and prefill with zeros."""
        self.state_buffer.clear()
        if self.use_vector_obs:
            z = torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device)
        else:
            z = torch.zeros(PREVIEW_SIZE, PREVIEW_SIZE, dtype=torch.float32, device=self.device)
        for _ in range(self.history_length):
            self.state_buffer.append(z.clone())

    def _read_leaderboard_score(self) -> Optional[int]:
        """Try reading the highlighted leaderboard score from the page."""
        if not self._page:
            return None
        try:
            item = self._page.query_selector("li.leaderboard__item--highlighted")
            if not item:
                return None
            node = item.query_selector(".leaderboard_score")
            if not node:
                return None
            txt = (node.inner_text() or "").strip()
            digits = "".join(ch for ch in txt if ch.isdigit())
            return int(digits) if digits else None
        except Exception:
            return None

    def _try_fast_replay(self) -> bool:
        """Try to click replay/reset buttons. Return True iff all clicks succeed."""
        if not self._page:
            return False
        try:
            clicker = ButtonClicker(self._page, self.cfg.click)
            ok_all = True
            for spec in self.cfg.replay_buttons:
                try:
                    ok = bool(clicker.click(spec))
                except Exception:
                    ok = False
                ok_all = ok_all and ok
            return ok_all
        except Exception:
            return False

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial stacked observation."""
        if not self._initialized:
            self._init_once()
        else:
            used_fast = self._try_fast_replay()
            if not used_fast:
                self._reinit_session()

        # Ensure canvas present (one retry via reinit if needed)
        for attempt in range(2):
            try:
                if not self._page:
                    self._reinit_session()
                self._canvas = self._page.wait_for_selector(self.cfg.canvas_selector, timeout=25_000, state="visible")
                break
            except Exception:
                if attempt == 0:
                    self._reinit_session()
                else:
                    logger.error("Failed to (re)acquire canvas.")
                    self._reset_buffer()
                    return torch.stack(list(self.state_buffer), 0)

        self._reset_buffer()
        self.steps_in_episode = 0

        # Initial observation
        x_norm = 0.5
        for attempt in range(2):
            bgr, alive = self._grab_canvas_bgr()
            if not alive:
                if attempt == 0:
                    self._reinit_session()
                    continue
                else:
                    bgr = np.zeros((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8)
                    break
            found, x_norm = detect_dot_x_norm(bgr)
            if not found:
                x_norm = 0.5
            break

        self._x_prev = float(x_norm)
        self._dx_prev = 0.0  # no motion known yet

        if self.use_vector_obs:
            obs = self._make_features(x_after=x_norm, dx_before=0.0, dx_after=0.0)
        elif self.synthetic_input:
            obs = render_dot_gray_84x84(x_norm).to(self.device)
        else:
            obs = self._to_84x84_gray(bgr).to(self.device)

        self.state_buffer.append(obs)

        # Preview only for image-based modes
        if self.display_input_window:
            color84 = self._to_84x84_bgr(bgr)
            self._show_combo(color84, obs, reward=0.0)

        return torch.stack(list(self.state_buffer), 0)

    # --------------------------------- Step -------------------------------- #

    def _is_out_of_bounds(self, x: float) -> bool:
        return (x < self.lower_limit_x) or (x > self.upper_limit_x)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Apply action, observe next state, compute reward, and flag done.

        Returns:
            (state_stack, reward, done)
        """
        self.steps_in_episode += 1

        # Pre-action snapshot
        x_before = float(self._x_prev)
        dx_before = float(self._dx_prev)

        # Act
        self._do_action(action)
        if self.display_agent_playing_window:
            self._show_combined_agent_view(action)

        if self._page:
            self._page.wait_for_timeout(self.step_wait_ms)

        # Observe
        bgr, alive = self._grab_canvas_bgr()
        if alive:
            found, x_after = detect_dot_x_norm(bgr)
            if not found:
                x_after = x_before
        else:
            x_after = x_before

        dx_after = float(x_after - x_before)
        obs_dot_gray = None
        # Build observation
        if self.use_vector_obs:
            obs = self._make_features(x_after=x_after, dx_before=dx_before, dx_after=dx_after)
            if self.display_input_window:
                obs_dot_gray = render_dot_gray_84x84(x_after).to(self.device)
        elif self.synthetic_input:
            obs = render_dot_gray_84x84(x_after).to(self.device)
        else:
            obs = self._to_84x84_gray(bgr).to(self.device)

        self.state_buffer.append(obs)

        # Termination and reward
        out_of_bounds = self._is_out_of_bounds(x_after)
        score = self._read_leaderboard_score() if alive else None
        if score is not None:
            self.last_score = score

        done = (not alive) or out_of_bounds or (score is not None)
        reward = -1.0 if done else center_reward_exp_triangle(float(x_after))

        # Update trackers for next step
        self._x_prev = float(x_after)
        self._dx_prev = float(dx_after)

        if self.display_input_window:
            color84 = self._to_84x84_bgr(bgr)
            self._show_combo(color84, obs_dot_gray, float(reward))

        return torch.stack(list(self.state_buffer), 0), float(reward), bool(done)
