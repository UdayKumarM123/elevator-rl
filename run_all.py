"""
run_all.py — One-Click Pipeline: Train + Evaluate + Plot + Git Tags
====================================================================
Runs the complete Smart Elevator RL pipeline end-to-end:

  Step 1: Install dependencies
  Step 2: Train V1 (500 episodes)
  Step 3: Train V2 (2000 episodes)
  Step 4: Evaluate baseline vs RL
  Step 5: Generate all plots
  Step 6: Print Git tagging instructions

Usage:
    python run_all.py
    python run_all.py --skip-v2       # skip long V2 training
    python run_all.py --eval-only     # only evaluate existing policies
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run(cmd, desc):
    banner(desc)
    print(f"  $ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ Done in {elapsed:.1f}s")
    else:
        print(f"\n  ✗ FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full Smart Elevator RL pipeline")
    parser.add_argument("--skip-v2",    action="store_true", help="Skip V2 (2000 ep) training")
    parser.add_argument("--eval-only",  action="store_true", help="Only run evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    args = parser.parse_args()

    total_start = time.time()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Smart Elevator Scheduling — RL + MLOps Pipeline    ║")
    print("║   SDG 11: Sustainable Cities & Communities           ║")
    print("║   SDG 9:  Industry, Innovation & Infrastructure      ║")
    print("╚══════════════════════════════════════════════════════╝")

    python = sys.executable

    if not args.eval_only:
        # ── Step 1: Install requirements ─────────────────────────────────────
        run([python, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            "Step 1/5 — Installing Dependencies")

        # ── Step 2: Train V1 ─────────────────────────────────────────────────
        run([python, "train.py", "--config", "configs/qlearning_v1.yaml"],
            "Step 2/5 — Training V1 (500 episodes, α=0.1, ε-decay=0.990)")

        # ── Step 3: Train V2 ─────────────────────────────────────────────────
        if not args.skip_v2:
            run([python, "train.py", "--config", "configs/qlearning_v2.yaml"],
                "Step 3/5 — Training V2 (2000 episodes, α=0.05, ε-decay=0.997)")
        else:
            banner("Step 3/5 — Skipping V2 training (--skip-v2 flag set)")

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    run([python, "evaluate.py", "--episodes", str(args.eval_episodes)],
        f"Step 4/5 — Evaluating: Baseline vs RL ({args.eval_episodes} episodes)")

    # ── Step 5: Generate Plots ────────────────────────────────────────────────
    run([python, "visualize.py"],
        "Step 5/5 — Generating Training Plots")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    banner("Pipeline Complete!")
    print(f"  Total time: {total_elapsed:.1f}s\n")
    print("  Files generated:")
    files_to_check = [
        "policies/policy_v1.pkl",
        "policies/policy_v2_explored.pkl",
        "experiments/results_1.csv",
        "experiments/results_2.csv",
        "experiments/log.json",
        "experiments/evaluation_results.json",
        "reports/figures/reward_curve.png",
        "reports/figures/training_dashboard.png",
        "reports/figures/wait_time_comparison.png",
        "reports/figures/metrics_bar_comparison.png",
    ]
    for f in files_to_check:
        exists = "✓" if Path(f).exists() else "✗ (missing)"
        print(f"    {exists}  {f}")

    print()
    print("  Next steps:")
    print("  ─────────────────────────────────────────────────────")
    print("  # Git tagging for MLOps versioning:")
    print("  git init                         # if not already a repo")
    print("  git add -A")
    print("  git commit -m 'initial: elevator RL project setup'")
    print("  git tag exp-qlearning-1          # after V1 training")
    print("  git tag exp-qlearning-2          # after V2 training")
    print()
    print("  # Live demo:")
    print("  python demo.py")
    print()
    print("  # View the final report:")
    print("  cat reports/final_report.md")
    print()


if __name__ == "__main__":
    main()
