"""
visualize.py - Generate Training Plots for Elevator RL Project
==============================================================
Reads results CSVs and produces:
  1. Reward curve over episodes (with smoothing)
  2. Epsilon decay over episodes
  3. Average waiting time over episodes
  4. Combined 2x2 dashboard

Usage:
    python visualize.py
    python visualize.py --csv experiments/results_1.csv --label "V1 (500 ep)"
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episode":       int(row["episode"]),
                "reward":        float(row["reward"]),
                "avg_wait_time": float(row["avg_wait_time"]),
                "epsilon":       float(row["epsilon"]),
                "deliveries":    int(row["deliveries"]),
                "energy_used":   int(row["energy_used"]),
            })
    return rows


def smooth(values, window=20):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i+1]) / (i - start + 1))
    return smoothed


def plot_all(datasets, labels, out_dir="reports/figures"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        colors = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]

        # 1. Reward over Episodes
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (data, label) in enumerate(zip(datasets, labels)):
            eps  = [r["episode"] for r in data]
            rews = [r["reward"]  for r in data]
            ax.plot(eps, rews, alpha=0.2, color=colors[i % len(colors)])
            ax.plot(eps, smooth(rews, 30), label=f"{label} (smoothed)",
                    color=colors[i % len(colors)], linewidth=2.2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total Reward", fontsize=12)
        ax.set_title("Average Reward over Episodes (Q-Learning)\n"
                     "SDG 11 - Sustainable Cities and Communities", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/reward_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Saved -> {out_dir}/reward_curve.png")

        # 2. Epsilon Decay
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (data, label) in enumerate(zip(datasets, labels)):
            ax.plot([r["episode"] for r in data],
                    [r["epsilon"] for r in data],
                    label=label, color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Epsilon", fontsize=12)
        ax.set_title("Exploration Rate (Epsilon) Decay over Training\n"
                     "e-greedy strategy: high exploration -> focused exploitation", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/epsilon_decay.png", dpi=150)
        plt.close(fig)
        print(f"  Saved -> {out_dir}/epsilon_decay.png")

        # 3. Average Wait Time
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, (data, label) in enumerate(zip(datasets, labels)):
            eps   = [r["episode"]       for r in data]
            waits = [r["avg_wait_time"] for r in data]
            ax.plot(eps, waits, alpha=0.2, color=colors[i % len(colors)])
            ax.plot(eps, smooth(waits, 30), label=f"{label} (smoothed)",
                    color=colors[i % len(colors)], linewidth=2.2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Avg Waiting Time (steps)", fontsize=12)
        ax.set_title("Passenger Avg Waiting Time during Training\n"
                     "SDG 11 - Reduction in Urban Congestion", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/wait_time_training.png", dpi=150)
        plt.close(fig)
        print(f"  Saved -> {out_dir}/wait_time_training.png")

        # 4. Dashboard 2x2
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        for i, (data, label) in enumerate(zip(datasets, labels)):
            eps      = [r["episode"]       for r in data]
            rews     = [r["reward"]        for r in data]
            waits    = [r["avg_wait_time"] for r in data]
            epsilons = [r["epsilon"]       for r in data]
            energy   = [r["energy_used"]   for r in data]
            c = colors[i % len(colors)]
            ax1.plot(eps, smooth(rews,   20), color=c, label=label, linewidth=1.8)
            ax2.plot(eps, smooth(waits,  20), color=c, label=label, linewidth=1.8)
            ax3.plot(eps, epsilons,           color=c, label=label, linewidth=1.8)
            ax4.plot(eps, smooth(energy, 20), color=c, label=label, linewidth=1.8)

        for ax, title, ylabel in [
            (ax1, "Reward Curve",         "Total Reward"),
            (ax2, "Avg Wait Time",        "Wait Time (steps)"),
            (ax3, "Epsilon Decay",        "Epsilon"),
            (ax4, "Energy Used/Episode",  "Moves"),
        ]:
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Episode", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

        fig.suptitle("Smart Elevator RL - Training Dashboard\n"
                     "SDG 11: Sustainable Cities  |  SDG 9: Industry and Infrastructure",
                     fontsize=13, y=1.01)
        fig.savefig(f"{out_dir}/training_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_dir}/training_dashboard.png")
        return True

    except ImportError:
        print("  [Note] matplotlib not installed - skipping plots.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate training plots")
    parser.add_argument("--csv",   nargs="+",
                        default=["experiments/results_1.csv",
                                 "experiments/results_2.csv"])
    parser.add_argument("--label", nargs="+",
                        default=["V1 (500 ep, alpha=0.1)",
                                 "V2 (2000 ep, alpha=0.05)"])
    parser.add_argument("--out",   default="reports/figures")
    args = parser.parse_args()

    print("=" * 50)
    print("  Smart Elevator RL - Plot Generator")
    print("=" * 50)

    datasets, labels = [], []
    for csv_path, label in zip(args.csv, args.label):
        if Path(csv_path).exists():
            data = read_csv(csv_path)
            datasets.append(data)
            labels.append(label)
            print(f"  Loaded {len(data)} episodes from {csv_path}")
        else:
            print(f"  [Skip] Not found: {csv_path}")

    if not datasets:
        print("  No data found. Run train.py first.")
        return

    plot_all(datasets, labels, args.out)
    print("\n  All plots saved to reports/figures/")


if __name__ == "__main__":
    main()
