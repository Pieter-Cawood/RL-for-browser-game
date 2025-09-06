from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Literal

from dotenv import load_dotenv


def _parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RainbowRLConfig:
    # ---- Experiment / device ----
    id: str = "browser-redbull-game"
    seed: int = 123
    disable_cuda: bool = False

    # ---- Rainbow hyperparams ----
    history_length: int = 3
    hidden_size: int = 128
    noisy_std: float = 0.1
    atoms: int = 51
    V_min: float = -1.0
    V_max: float = 1.0

    # Model path (depends on mode)
    model: Optional[str] = None

    # ---- Training/runtime ----
    memory_capacity: int = 100_000
    replay_frequency: int = 1
    priority_exponent: float = 0.5
    priority_weight: float = 0.4
    multi_step: int = 1
    discount: float = 0.995
    target_update: int = 1000
    learning_rate: float = 5e-4
    adam_eps: float = 1.5e-4
    batch_size: int = 64
    norm_clip: float = 10.0

    learn_start: int = 1_000
    learn_end: int = 300_000

    eval_interval: int = 0
    eval_size: int = 0
    checkpoint_interval: int = 1000
    memory: Optional[str] = None
    disable_bzip_memory: bool = False

    # ---- TensorBoard ----
    tb_dir: Optional[str] = None  # default: results/<id>/tb
    tb_image_interval: int = 0

    @staticmethod
    def from_env(mode: Literal["train", "test"] = "train") -> "RainbowRLConfig":
        """
        Load RainbowRLConfig from .env/environment variables.
        Mode determines which MODEL_* variable to prefer:
          - train: MODEL_TRAIN -> MODEL
          - test:  MODEL_TEST  -> MODEL
        """
        load_dotenv()

        def _get(k: str, default: Optional[str] = None) -> Optional[str]:
            v = os.getenv(k)
            return v if v is not None else default

        # --- Select model var based on mode ---
        model = None
        if mode == "train":
            model = _get("MODEL_TRAIN") or _get("MODEL")
        elif mode == "test":
            model = _get("MODEL_TEST") or _get("MODEL")

        return RainbowRLConfig(
            id=_get("RAINBOW_ID", "browser-redbull-game"),
            seed=int(_get("SEED", "123")),
            disable_cuda=_parse_bool(_get("DISABLE_CUDA"), False),

            history_length=int(_get("HISTORY_LENGTH", "3")),
            hidden_size=int(_get("HIDDEN_SIZE", "128")),
            noisy_std=float(_get("NOISY_STD", "0.1")),
            atoms=int(_get("ATOMS", "51")),
            V_min=float(_get("V_MIN", "-1.0")),
            V_max=float(_get("V_MAX", "1.0")),

            model=model,

            memory_capacity=int(_get("MEMORY_CAPACITY", "100000")),
            replay_frequency=int(_get("REPLAY_FREQUENCY", "1")),
            priority_exponent=float(_get("PRIORITY_EXPONENT", "0.5")),
            priority_weight=float(_get("PRIORITY_WEIGHT", "0.4")),
            multi_step=int(_get("MULTI_STEP", "1")),
            discount=float(_get("DISCOUNT", "0.995")),
            target_update=int(_get("TARGET_UPDATE", "1000")),
            learning_rate=float(_get("LEARNING_RATE", "0.0005")),
            adam_eps=float(_get("ADAM_EPS", "0.00015")),
            batch_size=int(_get("BATCH_SIZE", "64")),
            norm_clip=float(_get("NORM_CLIP", "10")),

            learn_start=int(_get("LEARN_START", "1000")),
            learn_end=int(_get("LEARN_END", "300000")),

            eval_interval=int(_get("EVAL_INTERVAL", "0")),
            eval_size=int(_get("EVAL_SIZE", "0")),
            checkpoint_interval=int(_get("CHECKPOINT_INTERVAL", "1000")),
            memory=_get("MEMORY"),
            disable_bzip_memory=_parse_bool(_get("DISABLE_BZIP_MEMORY"), False),

            tb_dir=_get("TB_DIR"),  # will fallback to results/<id>/tb in train
            tb_image_interval=int(_get("TB_IMAGE_INTERVAL", "0")),
        )
