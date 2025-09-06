"""
Train Rainbow on the browser redbull game environment with env-driven config.

Setup (from project root):
    pip install -r requirements.txt
    python -m playwright install chromium
    pip install -e .

Run (uses .env or environment variables):
    python -m src.scripts.train_agent

Warm-start:
    # set MODEL in .env to point at your checkpoint, or leave blank to train from scratch
    MODEL=results/browser-redbull-game/checkpoint.pth python -m src.scripts.train_agent

View training:
    tensorboard --logdir results/browser-redbull-game/tb
"""
from __future__ import annotations

import os
import time
import pickle
import bz2
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.app.browser_automation import BrowserGameConfig
from src.app.rainbow_rl import BrowserGameEnv, RainbowRLConfig, RainbowRLAgent, ReplayMemory


def log(s: str) -> None:
    print("[" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "] " + s)


def load_memory(memory_path: str, disable_bzip: bool):
    if disable_bzip:
        with open(memory_path, "rb") as f:
            return pickle.load(f)
    else:
        with bz2.open(memory_path, "rb") as f:
            return pickle.load(f)


def save_memory(memory, memory_path: str, disable_bzip: bool):
    if disable_bzip:
        with open(memory_path, "wb") as f:
            pickle.dump(memory, f)
    else:
        with bz2.open(memory_path, "wb") as f:
            pickle.dump(memory, f)


def resolve_near_script(path: Optional[str]) -> Optional[str]:
    """Return absolute, existing path for path. Try absolute, CWD, then script-dir."""
    if not path:
        return None
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.isfile(cwd_path):
        return cwd_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    near_script = os.path.join(script_dir, path)
    if os.path.isfile(near_script):
        return near_script
    return None


def main():
    # ---- Load env-driven training config ----
    cfg = RainbowRLConfig.from_env(mode="train")

    # ---- Device / seeds ----
    np.random.seed(cfg.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not cfg.disable_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(np.random.randint(1, 10000))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Resolve model path (accept cwd or script-dir relative)
    model_path = cfg.model
    if model_path:
        resolved = resolve_near_script(model_path)
        if resolved:
            model_path = resolved
        else:
            log(f"⚠️  MODEL '{model_path}' not found (checked CWD and script dir). Will train from scratch.")
            model_path = None

    # ---- Results dir & TensorBoard ----
    results_dir = os.path.join("results", cfg.id)
    os.makedirs(results_dir, exist_ok=True)
    tb_dir = cfg.tb_dir or os.path.join(results_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    log(f"TensorBoard logging to: {tb_dir}")

    # ---- Environment (browser-backed) ----
    app_cfg = BrowserGameConfig.from_env()
    env = BrowserGameEnv(
        app_cfg,
        device=device,
        feature_keys=("x", "dx_before", "dx_after"),
        history_length=cfg.history_length,
    )
    env.train()
    action_space = env.action_space()

    # ---- RainbowRLAgent ----
    # The RainbowRLAgent/ReplayMemory expect an 'args-like' object. We'll give them cfg via a thin shim.
    class _ArgsShim:
        pass

    args = _ArgsShim()
    # copy attributes RainbowRLAgent/ReplayMemory need
    for k, v in vars(cfg).items():
        setattr(args, k, v)
    setattr(args, "device", device)
    setattr(args, "model", model_path)

    dqn = RainbowRLAgent(args, env)  # RainbowRLAgent will load args.model if provided

    # ---- Replay memory ----
    if args.model and args.memory:
        if os.path.exists(args.memory):
            mem = load_memory(args.memory, args.disable_bzip_memory)
            log(f"Loaded replay memory from {args.memory}")
        else:
            log("Memory path provided but not found; starting fresh.")
            mem = ReplayMemory(args, args.memory_capacity, env)
    else:
        mem = ReplayMemory(args, args.memory_capacity, env)

    # ---- Training loop ----
    T_max = int(args.learn_end)
    priority_weight_increase = (1 - args.priority_weight) / max(1, (T_max - args.learn_start))

    dqn.train()
    dqn.reset_noise()
    done = True

    try:
        for T in trange(1, 1 + int(args.learn_end)):
            if done:
                state = env.reset()

            # NoisyNet exploration: reset noise periodically, act greedily under noisy weights
            if T % args.replay_frequency == 0:
                dqn.reset_noise()

            if (T < args.learn_start) and (args.model is None):
                action = np.random.randint(action_space)
            else:
                action = dqn.act(state)

            next_state, reward, done = env.step(action)

            mem.append(state, action, reward, done)


            # Optional: log a visualization of the latest stacked state
            if args.tb_image_interval and (T % args.tb_image_interval == 0):
                # state shape: [H, 84, 84]; stack is created when pushing to memory (RainbowRLAgent/env expect [history, 84, 84])
                # We can visualize the most recent frame only:
                img = state[-1:].unsqueeze(1)  # [1,1,84,84]
                writer.add_image("obs/gray84", img, global_step=T)

            if T >= args.learn_start:
                # PER annealing
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                # Learn at replay_frequency
                if T % args.replay_frequency == 0:
                    learn_stats = dqn.learn(mem)  # e.g., {"loss": float}
                    if learn_stats is not None:
                        if "loss" in learn_stats:
                            writer.add_scalar("train/loss", learn_stats["loss"], T)

                # Periodic quick eval on validation states (deterministic: disable noise)
                if (args.eval_interval != 0) and (T % args.eval_interval == 0):
                    dqn.eval()
                    done = False
                    state = env.reset()    
                    while not done:
                        # Act greedily (no noise)
                        action = dqn.act(state)
                        next_state, reward, done = env.step(action)
                        state = next_state
                    writer.add_scalar("env/last_score", getattr(env, "last_score", -1) or -1, T)
                    dqn.train()

                    if args.memory:
                        save_memory(mem, args.memory, args.disable_bzip_memory)

                # Update target
                if args.target_update and (T % args.target_update == 0):
                    dqn.update_target_net()

                # Checkpoint
                if args.checkpoint_interval and (T % args.checkpoint_interval == 0):
                    dqn.save(results_dir, "checkpoint.pth")

                # Log some basic counters each step
                writer.add_scalar("env/reward", float(reward), T)

            state = next_state

    finally:
        env.close()
        writer.close()
        dqn.save(results_dir, "final.pth")
        log(f"Saved final model to {os.path.join(results_dir, 'final.pth')}")
        log(f"TensorBoard logs in {tb_dir}")


if __name__ == "__main__":
    main()
