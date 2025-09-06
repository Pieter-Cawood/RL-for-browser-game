"""
Plot the exponential triangle reward function.

Run from the project root:
    python -m src.scripts.utils.test_reward_function

Prereqs:
    pip install -r requirements.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from src.app.rainbow_rl.env import center_reward_exp_triangle


def main() -> None:
    x_vals = np.linspace(0, 1, 500)
    ks = [1, 3, 6, 12]

    plt.figure(figsize=(8, 5))
    for k in ks:
        y_vals = [center_reward_exp_triangle(x, k) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f"k={k}")

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0.5, color="red", linestyle=":", linewidth=1)
    plt.xlabel("x_norm (guage position)")
    plt.ylabel("Reward")
    plt.title("Exponential Triangle Reward Function")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
