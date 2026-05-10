"""
demo.py — Interactive Live Demo: Baseline vs RL Policy
=======================================================
Runs an animated side-by-side comparison of the Nearest-Request
baseline and the trained Q-Learning policy in the terminal.

Usage:
    python demo.py                            # uses best available policy
    python demo.py --policy policies/policy_v1.pkl
    python demo.py --baseline-only            # watch baseline only
    python demo.py --rl-only                  # watch RL only
    python demo.py --delay 0.05               # faster animation
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sim.elevator_env import ElevatorEnv
from sim.visualizer import animate_episode
from agents.q_learning_agent import QLearningAgent
from agents.baseline_agent import NearestRequestBaseline


def make_env(seed=42):
    return ElevatorEnv(
        num_floors=10,
        max_capacity=8,
        episode_length=150,
        arrival_rate=0.3,
        seed=seed,
    )


def header(title):
    print("\033[2J\033[H", end="")  # clear
    print("\033[1m\033[96m" + "=" * 58 + "\033[0m")
    print(f"\033[1m\033[96m  {title}\033[0m")
    print("\033[2m  SDG 11: Sustainable Cities & Communities\033[0m")
    print("\033[1m\033[96m" + "=" * 58 + "\033[0m\n")


def print_comparison(b_stats, r_stats):
    print("\n\033[1m\033[96m" + "=" * 58 + "\033[0m")
    print("\033[1m  COMPARISON RESULTS\033[0m")
    print("=" * 58)
    metrics = [
        ("Avg Wait Time (steps)", "avg_wait_time",  True),
        ("Energy Used (moves)",   "energy_used",    True),
        ("Deliveries",            "deliveries",      False),
        ("Total Reward",          "total_reward",    False),
    ]
    print(f"  {'Metric':<26} {'Baseline':>12} {'RL Policy':>12} {'Winner':>8}")
    print("  " + "-" * 54)
    for label, key, lower_better in metrics:
        b_val = b_stats.get(key, 0)
        r_val = r_stats.get(key, 0)
        if lower_better:
            winner = "RL  ✓" if r_val < b_val else "BASE ✓" if b_val < r_val else "TIE"
        else:
            winner = "RL  ✓" if r_val > b_val else "BASE ✓" if b_val > r_val else "TIE"
        print(f"  {label:<26} {b_val:>12.2f} {r_val:>12.2f} {winner:>8}")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(description="Live Demo: Baseline vs RL Elevator")
    parser.add_argument("--policy",
                        default="policies/policy_v2_explored.pkl",
                        help="Path to trained RL policy .pkl")
    parser.add_argument("--delay", type=float, default=0.07,
                        help="Animation delay in seconds (default: 0.07)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--rl-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducible demo")
    args = parser.parse_args()

    # Load RL policy
    rl_agent = None
    policy_path = args.policy
    if not Path(policy_path).exists():
        policy_path = "policies/policy_v1.pkl"
    if Path(policy_path).exists():
        rl_agent = QLearningAgent.load(policy_path)
    else:
        if args.rl_only:
            print("No trained policy found. Run:")
            print("  python train.py --config configs/qlearning_v1.yaml")
            sys.exit(1)
        print("[Warning] No RL policy found — showing baseline only.")
        args.baseline_only = True

    baseline = NearestRequestBaseline()
    b_stats, r_stats = {}, {}

    # ── Run Baseline ────────────────────────────────────────────────────────
    if not args.rl_only:
        header("Nearest-Request BASELINE Controller")
        print("  Watch the traditional elevator controller in action...\n")
        time.sleep(1.5)
        env_b = make_env(seed=args.seed)
        b_stats = animate_episode(
            env=env_b,
            policy_fn=lambda e: baseline.select_action(e),
            policy_name="Nearest-Request Baseline",
            delay=args.delay,
        )
        print("\n  Press ENTER to watch the RL policy...")
        try:
            input()
        except EOFError:
            time.sleep(2)

    # ── Run RL Policy ────────────────────────────────────────────────────────
    if not args.baseline_only and rl_agent is not None:
        header("Q-Learning RL Policy Controller")
        print("  Watch the trained Reinforcement Learning agent in action...\n")
        time.sleep(1.5)
        env_r = make_env(seed=args.seed)
        state = env_r.reset()

        def rl_action(e):
            idx = e.state_to_index(e._get_state())
            return rl_agent.select_action(idx, training=False)

        r_stats = animate_episode(
            env=env_r,
            policy_fn=rl_action,
            policy_name="Q-Learning RL Policy",
            delay=args.delay,
        )

    # ── Comparison ───────────────────────────────────────────────────────────
    if b_stats and r_stats:
        print_comparison(b_stats, r_stats)

    print("\n  Demo complete!")
    print("  To reproduce: python demo.py --seed", args.seed)
    print("  To train more: python train.py --config configs/qlearning_v2.yaml")


if __name__ == "__main__":
    main()
