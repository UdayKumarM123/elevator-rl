"""
baseline_agent.py — Traditional Nearest-Request Elevator Controller.

This implements the classic "nearest call" (SCAN/SSTF) algorithm used in
conventional elevator systems. Used as the comparison baseline against
the RL-trained policy.

Strategy:
    1. If passengers are inside, head toward the nearest destination.
    2. Else, head toward the nearest floor with a waiting passenger.
    3. Open door when at a relevant floor.
    4. Stay idle if no requests.
"""

from typing import List, Optional, Tuple
from sim.elevator_env import ElevatorEnv, Action, DIR_UP, DIR_DOWN, DIR_IDLE
from sim.building import Building


class NearestRequestBaseline:
    """
    Nearest-Request (SSTF-style) elevator controller.

    At each step, decides the best action purely based on:
        - Passengers inside elevator (their destinations)
        - Floors with waiting passengers
    No learning involved — deterministic rule-based policy.
    """

    def select_action(self, env: ElevatorEnv) -> int:
        """
        Select next elevator action based on current environment state.

        Returns one of: MOVE_UP, MOVE_DOWN, STAY, OPEN_DOOR
        """
        current_floor = env.current_floor
        in_elevator   = env.passengers_in_elevator
        building      = env.building

        # ── Step 1: Drop off passengers at current floor ─────────────────────
        passengers_to_drop = [
            p for p in in_elevator if p.destination_floor == current_floor
        ]
        if passengers_to_drop:
            return Action.OPEN_DOOR

        # ── Step 2: Pick up passengers waiting on current floor ───────────────
        if building.floors[current_floor].has_request():
            # Only pick up if elevator not full
            if len(in_elevator) < env.max_capacity:
                return Action.OPEN_DOOR

        # ── Step 3: Move toward nearest target ───────────────────────────────
        target = self._get_nearest_target(env)

        if target is None:
            return Action.STAY

        if target > current_floor:
            return Action.MOVE_UP
        elif target < current_floor:
            return Action.MOVE_DOWN
        else:
            return Action.OPEN_DOOR

    def _get_nearest_target(self, env: ElevatorEnv) -> Optional[int]:
        """
        Find the nearest priority floor.

        Priority order:
            1. Destinations of passengers in elevator
            2. Floors with waiting passengers
        """
        current_floor = env.current_floor
        in_elevator   = env.passengers_in_elevator
        building      = env.building

        candidates: List[int] = []

        # Destinations of current passengers
        for p in in_elevator:
            candidates.append(p.destination_floor)

        # Floors with waiting passengers
        for floor_id in building.get_pending_requests():
            candidates.append(floor_id)

        if not candidates:
            return None

        # Nearest by absolute distance
        return min(candidates, key=lambda f: abs(f - current_floor))

    def run_episode(self, env: ElevatorEnv, render: bool = False) -> dict:
        """
        Run one complete episode using the baseline policy.

        Returns episode statistics.
        """
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action(env)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                print(env.render())
                print()

        stats = env.get_episode_stats()
        stats["policy"] = "nearest_request_baseline"
        return stats
