"""
evaluate.py - Baseline vs RL Policy Comparison
===============================================
Runs both the Nearest-Request Baseline and the trained Q-Learning policy
over the same simulator and prints a comparison table + generates plots.

Usage:
    python evaluate.py
    python evaluate.py --policy policies/policy_v2_explored.pkl --episodes 20

Final Evaluation (May 16):
    - Baseline vs RL table
    - Avg waiting time, energy usage, queue length
    - Plots saved to reports/figures/
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sim.elevator_env import ElevatorEnv
from agents.q_learning_agent import QLearningAgent
from agents.baseline_agent import NearestRequestBaseline


def make_env(seed=42):
    return ElevatorEnv(
        num_floors=10,
        max_capacity=8,
        episode_length=200,
        arrival_rate=0.3,
        seed=seed,
    )


def run_rl_episode(env, agent):
    state = env.reset()
    idx = env.state_to_index(state)
    done = False
    queue_lengths = []
    while not done:
        action = agent.select_action(idx, training=False)
        next_state, reward, done, info = env.step(action)
        idx = env.state_to_index(next_state)
        queue_lengths.append(info["waiting"])
    stats = env.get_episode_stats()
    stats["avg_queue"] = float(np.mean(queue_lengths))
    return stats, queue_lengths


def run_baseline_episode(env, baseline):
    state = env.reset()
    done = False
    queue_lengths = []
    while not done:
        action = baseline.select_action(env)
        next_state, reward, done, info = env.step(action)
        queue_lengths.append(info["waiting"])
    stats = env.get_episode_stats()
    stats["avg_queue"] = float(np.mean(queue_lengths))
    return stats, queue_lengths


def aggregate(all_stats):
    keys = ["total_reward", "avg_wait_time", "energy_used", "deliveries", "avg_queue"]
    agg = {}
    for k in keys:
        vals = [s[k] for s in all_stats]
        agg[k] = {"mean": round(float(np.mean(vals)), 2),
                  "std":  round(float(np.std(vals)),  2)}
    return agg


def print_table(baseline_agg, rl_agg):
    metrics = [
        ("Avg Wait Time (steps)", "avg_wait_time", True),
        ("Avg Queue Length",      "avg_queue",      True),
        ("Energy (moves/ep)",     "energy_used",    True),
        ("Deliveries/ep",         "deliveries",     False),
        ("Total Reward",          "total_reward",   False),
    ]
    sep = "=" * 68
    print("\n" + sep)
    print("  BASELINE vs RL COMPARISON TABLE  (Final Evaluation - May 16)")
    print("  SDG 11: Sustainable Cities and Communities")
    print(sep)
    print(f"  {'Metric':<28} {'Nearest-Request':>16} {'Q-Learning RL':>14}  {'Result':>8}")
    print("  " + "-" * 62)
    for label, key, lower_better in metrics:
        b = baseline_agg[key]["mean"]
        r = rl_agg[key]["mean"]
        if lower_better:
            imp    = ((b - r) / b) * 100 if b != 0 else 0
            winner = "[RL WINS]" if r < b else ("[BASE]" if b < r else "[TIE]")
        else:
            imp    = ((r - b) / abs(b)) * 100 if b != 0 else 0
            winner = "[RL WINS]" if r > b else ("[BASE]" if b > r else "[TIE]")
        print(f"  {label:<28} {b:>16.2f} {r:>14.2f}  {winner:>9}  {imp:+.1f}%")
    print(sep)


def try_plot(baseline_stats_list, rl_stats_list, baseline_ql_list, rl_ql_list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = Path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        episodes = list(range(1, len(baseline_stats_list) + 1))

        # Plot 1: Avg Wait Time per Episode
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, [s["avg_wait_time"] for s in baseline_stats_list],
                label="Nearest-Request Baseline", color="#e74c3c",
                linewidth=2, marker="o", markersize=4)
        ax.plot(episodes, [s["avg_wait_time"] for s in rl_stats_list],
                label="Q-Learning RL Policy", color="#2ecc71",
                linewidth=2, marker="s", markersize=4)
        ax.set_xlabel("Evaluation Episode", fontsize=12)
        ax.set_ylabel("Average Waiting Time (steps)", fontsize=12)
        ax.set_title("Baseline vs RL: Average Passenger Waiting Time\n"
                     "SDG 11 - Sustainable Cities and Communities", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "wait_time_comparison.png", dpi=150)
        plt.close(fig)
        print("  Plot saved -> reports/figures/wait_time_comparison.png")

        # Plot 2: Queue Length over time (first episode)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list(range(len(baseline_ql_list[0]))), baseline_ql_list[0],
                label="Nearest-Request Baseline", color="#e74c3c", alpha=0.8, linewidth=1.5)
        ax.plot(list(range(len(rl_ql_list[0]))), rl_ql_list[0],
                label="Q-Learning RL Policy", color="#2ecc71", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Step (within episode)", fontsize=12)
        ax.set_ylabel("Total Waiting Passengers", fontsize=12)
        ax.set_title("Queue Length Over Time (Episode 1)\nBaseline vs RL Policy", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "queue_length_comparison.png", dpi=150)
        plt.close(fig)
        print("  Plot saved -> reports/figures/queue_length_comparison.png")

        # Plot 3: Bar chart comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        metric_keys   = ["avg_wait_time", "energy_used", "avg_queue"]
        metric_labels = ["Avg Wait Time", "Energy (moves)", "Avg Queue"]
        b_vals = [np.mean([s[k] for s in baseline_stats_list]) for k in metric_keys]
        r_vals = [np.mean([s[k] for s in rl_stats_list])       for k in metric_keys]
        x = np.arange(len(metric_labels))
        w = 0.35
        bars_b = ax.bar(x - w/2, b_vals, w, label="Baseline",   color="#e74c3c", alpha=0.85)
        bars_r = ax.bar(x + w/2, r_vals, w, label="Q-Learning", color="#2ecc71", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Performance Metrics: Baseline vs RL  (Lower is Better)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar in list(bars_b) + list(bars_r):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / "metrics_bar_comparison.png", dpi=150)
        plt.close(fig)
        print("  Plot saved -> reports/figures/metrics_bar_comparison.png")

        return True
    except ImportError:
        print("  [Note] matplotlib not installed - skipping plots.")
        return False


def save_eval_json(baseline_agg, rl_agg, n_episodes):
    Path("experiments").mkdir(exist_ok=True)
    result = {
        "evaluation_episodes": n_episodes,
        "baseline":  baseline_agg,
        "rl_policy": rl_agg,
    }
    with open("experiments/evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  Eval JSON -> experiments/evaluation_results.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare Baseline vs RL")
    parser.add_argument("--policy",          default="policies/policy_v2_explored.pkl")
    parser.add_argument("--fallback_policy", default="policies/policy_v1.pkl")
    parser.add_argument("--episodes",        type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("  Smart Elevator RL - Evaluation and Comparison")
    print("  SDG 11: Sustainable Cities and Communities")
    print("=" * 60)

    policy_path = args.policy
    if not Path(policy_path).exists():
        print(f"  Policy not found: {policy_path}, trying fallback...")
        policy_path = args.fallback_policy
    if not Path(policy_path).exists():
        print("  No trained policy found. Run:")
        print("    python train.py --config configs/qlearning_v1.yaml")
        sys.exit(1)

    agent    = QLearningAgent.load(policy_path)
    baseline = NearestRequestBaseline()

    baseline_all_stats, rl_all_stats = [], []
    baseline_ql_list,   rl_ql_list   = [], []

    print(f"\n  Running {args.episodes} evaluation episodes...")
    print(f"  {'Ep':>4}  {'Baseline Wait':>14}  {'RL Wait':>10}  {'Better':>8}")
    print("  " + "-" * 42)
    for ep in range(args.episodes):
        seed  = 100 + ep
        env_b = make_env(seed=seed)
        b_stats, b_ql = run_baseline_episode(env_b, baseline)
        baseline_all_stats.append(b_stats)
        baseline_ql_list.append(b_ql)

        env_r = make_env(seed=seed)
        r_stats, r_ql = run_rl_episode(env_r, agent)
        rl_all_stats.append(r_stats)
        rl_ql_list.append(r_ql)

        bw = b_stats["avg_wait_time"]
        rw = r_stats["avg_wait_time"]
        better = "RL" if rw < bw else ("BASE" if bw < rw else "TIE")
        print(f"  {ep+1:>4}  {bw:>14.2f}  {rw:>10.2f}  {better:>8}")

    baseline_agg = aggregate(baseline_all_stats)
    rl_agg       = aggregate(rl_all_stats)

    print_table(baseline_agg, rl_agg)
    try_plot(baseline_all_stats, rl_all_stats, baseline_ql_list, rl_ql_list)
    save_eval_json(baseline_agg, rl_agg, args.episodes)

    # SDG Impact
    wait_b     = baseline_agg["avg_wait_time"]["mean"]
    wait_r     = rl_agg["avg_wait_time"]["mean"]
    reduction  = ((wait_b - wait_r) / wait_b) * 100 if wait_b > 0 else 0
    energy_b   = baseline_agg["energy_used"]["mean"]
    energy_r   = rl_agg["energy_used"]["mean"]
    energy_red = ((energy_b - energy_r) / energy_b) * 100 if energy_b > 0 else 0

    print("\n  SDG Impact Statement:")
    print(f"  Reducing avg wait-time by {reduction:.1f}% supports SDG 11 by")
    print("  reducing congestion, fuel waste, and air pollution in smart buildings.")
    print(f"  Energy reduction of {energy_red:.1f}% supports SDG 9 by advancing")
    print("  infrastructure efficiency through AI-driven automation.")
    print()


if __name__ == "__main__":
    main()
