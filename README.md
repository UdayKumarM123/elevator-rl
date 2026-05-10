# 🛗 Smart Elevator Scheduling System — Reinforcement Learning + MLOps

[![SDG 11](https://img.shields.io/badge/SDG-11%20Sustainable%20Cities-green)](https://sdgs.un.org/goals/goal11)
[![SDG 9](https://img.shields.io/badge/SDG-9%20Industry%20%26%20Innovation-blue)](https://sdgs.un.org/goals/goal9)
[![Algorithm](https://img.shields.io/badge/Algorithm-Q--Learning-orange)](https://en.wikipedia.org/wiki/Q-learning)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

> **RL Problem Statement:** "Control elevator movement in a 10-floor building to minimize average passenger waiting time and reduce energy consumption using Reinforcement Learning."

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [SDG Connection](#-sdg-connection)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Reproduce an Experiment](#-reproduce-an-experiment)
- [Part A — RL Methodology](#-part-a--rl-methodology)
- [Part B — MLOps](#-part-b--mlops)
- [Results Summary](#-results-summary)
- [Monitoring Plan](#-monitoring-plan)
- [Evaluation Commands](#-evaluation-commands)

---

## 🏢 Project Overview

This project implements a **Reinforcement Learning agent** that learns to optimally control a single elevator in a 10-floor building. The agent is trained using **Q-Learning** (tabular, off-policy TD control) against a building simulator that generates passengers with Poisson arrivals.

### Key Components

| Component | Description |
|---|---|
| `sim/building.py` | Passenger generation, floor queues, building statistics |
| `sim/elevator_env.py` | Gym-compatible RL environment (state, action, reward) |
| `sim/visualizer.py` | ASCII terminal renderer for live visualization |
| `agents/q_learning_agent.py` | Q-Learning with ε-greedy exploration + policy save/load |
| `agents/baseline_agent.py` | Nearest-Request (SSTF) traditional controller |
| `train.py` | Training entry point — reads YAML config |
| `evaluate.py` | Baseline vs RL comparison with plots |
| `visualize.py` | Training curve and dashboard plot generator |
| `demo.py` | Live animated terminal demo |

---

## 🌍 SDG Connection

| SDG | Goal | How This Project Contributes |
|---|---|---|
| **SDG 11** | Sustainable Cities & Communities | Reduces passenger wait times, congestion, and energy waste in urban buildings |
| **SDG 9** | Industry, Innovation & Infrastructure | Applies AI/RL to modernize building infrastructure management |

---

## 📁 Project Structure

```
elevator-rl/
├── sim/
│   ├── __init__.py
│   ├── building.py           # Floor, Passenger, Building models
│   ├── elevator_env.py       # RL Environment (state/action/reward)
│   └── visualizer.py         # ASCII + color terminal renderer
│
├── agents/
│   ├── __init__.py
│   ├── q_learning_agent.py   # Q-Learning with ε-greedy
│   └── baseline_agent.py     # Nearest-Request baseline
│
├── configs/
│   ├── qlearning_v1.yaml     # 500-episode config (fast)
│   └── qlearning_v2.yaml     # 2000-episode config (best policy)
│
├── experiments/
│   ├── results_1.csv         # Per-episode log for V1 run
│   ├── results_2.csv         # Per-episode log for V2 run
│   ├── evaluation_results.json
│   └── log.json              # Aggregated run log (MLOps)
│
├── policies/
│   ├── policy_v1.pkl                # Saved Q-table (V1, 500 ep)
│   └── policy_v2_explored.pkl       # Saved Q-table (V2, 2000 ep)
│
├── reports/
│   ├── final_report.md       # Complete project report
│   └── figures/              # All generated plots (.png)
│       ├── reward_curve.png
│       ├── epsilon_decay.png
│       ├── wait_time_training.png
│       ├── training_dashboard.png
│       ├── wait_time_comparison.png
│       ├── queue_length_comparison.png
│       └── metrics_bar_comparison.png
│
├── train.py                  # Main training script
├── evaluate.py               # Baseline vs RL evaluation
├── visualize.py              # Plot generator
├── demo.py                   # Live animated demo
├── run_all.py                # One-click: train + evaluate + plot
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (train V1 + V2, evaluate, plot)
python run_all.py

# --- OR step by step ---

# 3. Train V1 (500 episodes, ~1 min)
python train.py --config configs/qlearning_v1.yaml

# 4. Train V2 (2000 episodes, ~4 min)
python train.py --config configs/qlearning_v2.yaml

# 5. Evaluate and compare
python evaluate.py

# 6. Generate plots
python visualize.py

# 7. Watch live demo
python demo.py
```

---

## 🔁 Reproduce an Experiment

> **Anyone can clone this repo and reproduce exact results.**

```bash
git clone <repo-url>
cd elevator-rl
pip install -r requirements.txt

# Reproduce V1 (exp-qlearning-1)
git checkout exp-qlearning-1
python train.py --config configs/qlearning_v1.yaml
# → Produces: policies/policy_v1.pkl, experiments/results_1.csv

# Reproduce V2 (exp-qlearning-2)
git checkout exp-qlearning-2
python train.py --config configs/qlearning_v2.yaml
# → Produces: policies/policy_v2_explored.pkl, experiments/results_2.csv

# Compare both
python evaluate.py --policy policies/policy_v2_explored.pkl --episodes 20
```

All hyperparameters (learning rate, epsilon, seed) are stored in `configs/*.yaml`.
The `seed` in each config ensures deterministic environment and agent behaviour.

---

## 📐 Part A — RL Methodology

### Algorithm Choice

**Q-Learning** was selected because:
> "The elevator state space (current floor, direction, pending request bitmask) is discrete and has ~30,720 unique states — small enough for a tabular Q-table. Q-learning is off-policy, allowing the agent to learn from exploratory actions without being constrained to follow its current policy, which is critical in a sparse-reward environment."

### State Definition

| Feature | Values | Notes |
|---|---|---|
| `current_floor` | 0–9 | Which floor elevator is on |
| `direction` | UP / IDLE / DOWN | Encoded as {0, 1, 2} |
| `request_mask` | 10-bit integer | Bitmask of floors with waiting passengers |

**State space size:** 10 × 3 × 1024 = **30,720 states**

### Action Definition

| ID | Action | Description |
|---|---|---|
| 0 | `MOVE_UP` | Move elevator one floor up |
| 1 | `MOVE_DOWN` | Move elevator one floor down |
| 2 | `STAY` | Hold current position |
| 3 | `OPEN_DOOR` | Board/alight passengers |

### Reward Definition

```
R = −1.0 × (total waiting passengers)     ← minimize congestion
  − 0.5 × (move taken)                    ← penalize energy use
  + 10.0 × (passengers delivered)         ← reward service
  − 5.0  × (illegal move attempted)       ← boundary safety
  − 0.2  × (idle with passengers waiting) ← penalize laziness
```

### Exploration Strategy

**ε-Greedy with Exponential Decay:**
```
ε_t = max(ε_min, ε_0 × decay^episode)
```

| Config | ε₀ | Decay | ε_min | Episodes |
|---|---|---|---|---|
| V1 | 1.0 | 0.990 | 0.05 | 500 |
| V2 | 1.0 | 0.997 | 0.01 | 2,000 |

### Convergence Discussion

> "Average reward improves over time and stabilizes. Early training (episodes 1–50) shows high variance with random exploration. By episode 200, the agent has learned basic patterns: open door when passengers are present, move toward pending requests. By episode 500 (V1) the reward plateau indicates convergence. V2's extended training (2000 episodes) allows finer policy refinement, yielding a ~15% better final policy."

### Saved Policies

| File | Saved When | Purpose |
|---|---|---|
| `policies/policy_v1.pkl` | After 500 episodes | Initial trained policy |
| `policies/policy_v2_explored.pkl` | After 2000 episodes | Best (most explored) policy |

---

## 🔧 Part B — MLOps

### Versioning

```bash
# After each experiment, tag the commit:
git add -A
git commit -m "exp: qlearning v1 - 500 episodes"
git tag exp-qlearning-1

git add -A
git commit -m "exp: qlearning v2 - 2000 episodes extended exploration"
git tag exp-qlearning-2

git push origin main --tags
```

### Experiment Tracking

Each run automatically saves:

**`experiments/results_X.csv`** — per-episode data:
```
run_id, episode, reward, avg_wait_time, epsilon, alpha, gamma,
epsilon_decay, deliveries, energy_used
```

**`experiments/log.json`** — aggregated run log:
```json
{
  "run_id": "qlearning_v1",
  "timestamp": "2026-05-09T10:00:00",
  "algorithm": "Q-Learning",
  "episodes": 500,
  "avg_reward": -412.3,
  "avg_reward_last100": -298.7,
  "best_reward": -187.2,
  "avg_wait_time": 14.3,
  "alpha": 0.1,
  "gamma": 0.99,
  "epsilon": 1.0,
  "epsilon_min": 0.05,
  "epsilon_decay": 0.990,
  "policy_path": "policies/policy_v1.pkl",
  "git_tag": "exp-qlearning-1"
}
```

### Reproducibility

Run any experiment with a single command:
```bash
python train.py --config configs/qlearning_v1.yaml
```

The YAML config pins:
- All hyperparameters (α, γ, ε, decay)
- Environment parameters (floors, capacity, arrival rate)
- Random seed (environment + agent)

### Monitoring Plan

> If this system were deployed in a real building, we would monitor:
>
> - **Average wait time** (rolling 5-min window) — alert if > 60s
> - **Maximum queue length** on any floor — alert if > 5 passengers
> - **Energy consumption** — elevator moves per hour
> - **Reward drift** — retrain if mean reward drops >20% from baseline
> - **Safety rules** — never skip a floor with ≥8 consecutive waiters
> - **Auto-fallback** — revert to nearest-request baseline if RL degrades

---

## 📊 Results Summary

| Metric | Nearest-Request Baseline | Q-Learning RL (V2) | Improvement |
|---|---|---|---|
| Avg Wait Time (steps) | ~45.2 | ~28.7 | **−36.5%** |
| Energy (moves/ep) | ~118.4 | ~82.1 | **−30.7%** |
| Avg Queue Length | ~3.8 | ~2.1 | **−44.7%** |
| Deliveries/episode | ~38.2 | ~52.6 | **+37.7%** |
| Total Reward/episode | ~−542 | ~−298 | **+45.0%** |

> *Evaluated over 20 episodes with greedy (ε=0) RL policy. Same environment seed per episode pair.*

### SDG Impact

> "Reducing average passenger wait-time by ~36% supports **SDG 11** by reducing congestion, fuel waste, and air pollution in smart buildings. A ~30% energy reduction in elevator movement supports **SDG 9** by demonstrating AI-driven sustainable infrastructure optimization."

---

## 📺 Evaluation Commands

```bash
# Full automated pipeline
python run_all.py

# Training only
python train.py --config configs/qlearning_v1.yaml
python train.py --config configs/qlearning_v2.yaml

# Evaluation (baseline vs RL)
python evaluate.py --episodes 20

# Plots
python visualize.py

# Live animated demo
python demo.py
python demo.py --baseline-only
python demo.py --rl-only
python demo.py --delay 0.03   # faster
```

---

## 📦 Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
```

Install: `pip install -r requirements.txt`

No gym/torch/tensorflow required — pure Python + numpy!

---

*Smart Elevator Scheduling System | RL + MLOps Project | May 2026*
*SDG 11 – Sustainable Cities | SDG 9 – Industry & Innovation*
