"""
Run a trained Rainbow agent in the browser canvas ONCE, then pause so you can
enter your details in the page. Press ENTER in the terminal to close.

Configure via .env or environment variables:

Required/Useful:
    MODEL_TEST=results/browser-redbull-game/final.pth  # preferred for this script, can use checkpoint.pth too 
    # (falls back to MODEL if MODEL_TEST is not set)

Optional:
    HEADLESS=false
    MAX_STEPS=100000

To run:
    python -m src.scripts.run_agent
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import torch
from dotenv import load_dotenv

from src.app.browser_automation import BrowserGameConfig
from src.app.rainbow_rl import BrowserGameEnv, RainbowRLConfig, RainbowRLAgent


def log(s: str) -> None:
    print("[" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "] " + s)


def _get(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if v is not None else default


def _parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_near_script(path: str) -> str:
    """Return absolute, existing path for path. Try absolute, CWD, then script-dir."""
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.isfile(cwd_path):
        return cwd_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    near_script = os.path.join(script_dir, path)
    if os.path.isfile(near_script):
        return near_script
    raise FileNotFoundError(f"Model checkpoint not found: {path}")


def main():
    # Load .env first
    load_dotenv()

    # Get test-mode training config (selects MODEL_TEST -> MODEL)
    cfg = RainbowRLConfig.from_env(mode="test")

    # Runtime-only flags
    headless = _parse_bool(_get("HEADLESS"), False)
    max_steps = int(_get("MAX_STEPS", "1000000"))

    # Device
    if torch.cuda.is_available() and not cfg.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Resolve model path (prefer MODEL_TEST via cfg.model; fallback to a common default)
    model_path = cfg.model or "results/browser-redbull-game/final.pth"
    model_path = resolve_near_script(model_path)

    # Prepare args shim for RainbowRLAgent compatibility
    class _ArgsShim:
        pass

    args = _ArgsShim()
    for k, v in vars(cfg).items():
        setattr(args, k, v)
    setattr(args, "device", device)
    setattr(args, "model", model_path)

    # Browser/env config
    # BrowserGameConfig.from_env reads HEADLESS already; ensure env reflects our flag.
    if headless:
        os.environ["HEADLESS"] = "true"
    app_cfg = BrowserGameConfig.from_env()

    env = BrowserGameEnv(app_cfg, device=device, history_length=cfg.history_length)
    env.eval()  # eval mode

    # RainbowRLAgent (loads checkpoint at args.model)
    agent = RainbowRLAgent(args, env)
    agent.eval()

    print(f"Loaded model from {model_path}")
    if headless:
        print("Running headless (you won't be able to type into the page).")
    else:
        print("Running with a visible browser. After the run ends, fill in your details on the page,")
        print("then return here and press ENTER to close.")

    # Single episode run
    try:
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            action = agent.act(state)  # greedy / deterministic
            next_state, reward, done = env.step(action)
            total_reward += float(reward)
            steps += 1
            state = next_state

        print(f"\nEpisode finished: steps={steps}, total_reward={total_reward:.3f}")

        # Keep browser open so you can type on the page
        if not headless:
            try:
                input("\nYou can now enter your details in the browser. Press ENTER here when you're done to close...")
            except EOFError:
                # If no TTY, just give a short grace period
                import time as _t
                _t.sleep(5)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
