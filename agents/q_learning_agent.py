"""
q_learning_agent.py — Tabular Q-Learning Agent with ε-greedy exploration.

Algorithm: Q-Learning (off-policy TD control)
Why chosen: The elevator state space (floor, direction, request bitmask) is
            discrete and manageable for a tabular approach, making Q-learning
            ideal — interpretable, fast, and provably convergent.

Update rule:
    Q(s,a) ← Q(s,a) + α [r + γ·max_a' Q(s',a') - Q(s,a)]

Exploration: ε-greedy with exponential decay
    ε_t = max(ε_min, ε_0 · decay^t)
"""

import numpy as np
import pickle
from typing import Tuple, Optional, List, Dict
from pathlib import Path


class QLearningAgent:
    """
    Tabular Q-Learning Agent for the Elevator Scheduling problem.

    Attributes:
        alpha   (float): Learning rate [0, 1].
        gamma   (float): Discount factor [0, 1].
        epsilon (float): Exploration rate (decays over time).
        epsilon_decay (float): Multiplicative decay per episode.
        epsilon_min   (float): Lower bound on epsilon.
        n_actions     (int):   Number of discrete actions.
        state_size    (int):   Total number of discrete states.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = 42,
    ):
        self.state_size    = state_size
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.rng = np.random.RandomState(seed)

        # Q-table: shape [state_size, n_actions], initialized to zeros
        self.q_table = np.zeros((state_size, n_actions), dtype=np.float32)

        # Training history
        self.episode_rewards: List[float] = []
        self.episode_epsilons: List[float] = []
        self.episode_waits: List[float] = []

    # ── Action Selection ────────────────────────────────────────────────────

    def select_action(self, state_index: int, training: bool = True) -> int:
        """
        ε-greedy action selection.

        During training: explore with probability ε, exploit otherwise.
        During evaluation: always exploit (greedy).
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[state_index]))

    # ── Q-Table Update ──────────────────────────────────────────────────────

    def update(
        self,
        state_index: int,
        action: int,
        reward: float,
        next_state_index: int,
        done: bool,
    ):
        """
        Q-Learning update (off-policy TD).

        Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',·) - Q(s,a)]
        """
        current_q = self.q_table[state_index, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_index])

        td_error = target - current_q
        self.q_table[state_index, action] += self.alpha * td_error

    # ── Epsilon Decay ────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Decay epsilon after each episode (exponential schedule)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the Q-table and hyperparameters to a .pkl file."""
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_size": self.state_size,
            "n_actions": self.n_actions,
            "episode_rewards": self.episode_rewards,
            "episode_epsilons": self.episode_epsilons,
            "episode_waits": self.episode_waits,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [OK] Policy saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load a previously saved agent from a .pkl file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            state_size=payload["state_size"],
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
        )
        agent.q_table = payload["q_table"]
        agent.episode_rewards = payload.get("episode_rewards", [])
        agent.episode_epsilons = payload.get("episode_epsilons", [])
        agent.episode_waits = payload.get("episode_waits", [])
        print(f"  [OK] Policy loaded <- {path}")
        return agent

    # -- Statistics -----------------------------------------------------------

    def record_episode(self, total_reward: float, avg_wait: float):
        """Record end-of-episode statistics."""
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)
        self.episode_waits.append(avg_wait)

    def get_stats(self) -> Dict:
        if not self.episode_rewards:
            return {}
        rewards = np.array(self.episode_rewards)
        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward_all": float(np.mean(rewards)),
            "avg_reward_last100": float(np.mean(last_100)),
            "best_reward": float(np.max(rewards)),
            "current_epsilon": self.epsilon,
            "avg_wait_last100": float(np.mean(self.episode_waits[-100:])),
        }

    def __repr__(self):
        return (
            f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"eps={self.epsilon:.4f}, states={self.state_size}, actions={self.n_actions})"
        )
