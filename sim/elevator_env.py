"""
elevator_env.py -- RL Environment for Smart Elevator Scheduling System.

State:  (current_floor, direction, nearest_request_floor, passengers_onboard)
Action: MOVE_UP / MOVE_DOWN / STAY / OPEN_DOOR
Reward: -wait_penalty + delivery_bonus - energy_penalty

This uses a compact, discrete state encoding suitable for tabular Q-Learning.

SDG 11 - Sustainable Cities and Communities
SDG 9  - Industry, Innovation and Infrastructure
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from enum import IntEnum

from sim.building import Building, Passenger


class Action(IntEnum):
    MOVE_UP   = 0
    MOVE_DOWN = 1
    STAY      = 2
    OPEN_DOOR = 3


ACTION_NAMES = {
    Action.MOVE_UP:   "MOVE_UP",
    Action.MOVE_DOWN: "MOVE_DOWN",
    Action.STAY:      "STAY",
    Action.OPEN_DOOR: "OPEN_DOOR",
}

DIR_UP   =  1
DIR_DOWN = -1
DIR_IDLE =  0


class ElevatorEnv:
    """
    Smart Elevator RL Environment (Gym-style interface).

    State representation (compact for Q-learning):
        - current_floor       : 0..num_floors-1
        - direction           : {0=idle, 1=up, 2=down}
        - nearest_req_floor   : 0..num_floors  (num_floors = no request)
        - passengers_onboard  : 0..max_capacity (bucketed into 3: 0,1-3,4+)

    Total states = 10 * 3 * 11 * 3 = 990  (very tractable for Q-table)

    Actions:
        0 = MOVE_UP
        1 = MOVE_DOWN
        2 = STAY
        3 = OPEN_DOOR

    Reward:
        +10  per passenger delivered
        -0.5 per move (energy penalty)
        -1.0 per waiting passenger per step (scaled down for stability)
        -5.0 for illegal moves
    """

    WAIT_PENALTY    = -0.5
    ENERGY_PENALTY  = -0.3
    DELIVERY_BONUS  = +10.0
    ILLEGAL_PENALTY = -5.0

    def __init__(
        self,
        num_floors: int = 10,
        max_capacity: int = 8,
        episode_length: int = 200,
        arrival_rate: float = 0.3,
        seed: int = 42,
    ):
        self.num_floors     = num_floors
        self.max_capacity   = max_capacity
        self.episode_length = episode_length
        self.arrival_rate   = arrival_rate
        self.seed           = seed

        self.building = Building(
            num_floors=num_floors,
            arrival_rate=arrival_rate,
            seed=seed,
        )

        # Elevator state
        self.current_floor: int = 0
        self.direction: int = DIR_IDLE
        self.passengers_in_elevator: List[Passenger] = []
        self.door_open: bool = False

        # Episode tracking
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.energy_used: int = 0
        self.deliveries: int = 0

        # Action space
        self.n_actions = len(Action)

        # State space dimensions (compact):
        self.n_floors      = num_floors           # 10
        self.n_directions  = 3                    # idle/up/down
        self.n_req_states  = num_floors + 1       # 0..9 + "none"
        self.n_load_states = 3                    # 0, 1-3, 4+

    # ------------------------------------------------------------------

    def reset(self) -> Tuple:
        """Reset environment; return initial state."""
        self.building.reset()
        self.current_floor = 0
        self.direction = DIR_IDLE
        self.passengers_in_elevator = []
        self.door_open = False
        self.step_count = 0
        self.total_reward = 0.0
        self.energy_used = 0
        self.deliveries = 0
        self.building.step()
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """Execute action, advance one step."""
        self.step_count += 1
        reward = 0.0
        illegal = False

        # -- Execute action --
        if action == Action.MOVE_UP:
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
                self.direction = DIR_UP
                self.energy_used += 1
                reward += self.ENERGY_PENALTY
            else:
                illegal = True

        elif action == Action.MOVE_DOWN:
            if self.current_floor > 0:
                self.current_floor -= 1
                self.direction = DIR_DOWN
                self.energy_used += 1
                reward += self.ENERGY_PENALTY
            else:
                illegal = True

        elif action == Action.STAY:
            self.direction = DIR_IDLE

        elif action == Action.OPEN_DOOR:
            picked  = self._pickup_passengers()
            dropped = self._dropoff_passengers()
            if picked == 0 and dropped == 0:
                reward -= 0.5  # useless door open

        if illegal:
            reward += self.ILLEGAL_PENALTY

        # -- Delivery bonus --
        delivered_this_step = self._count_new_deliveries()
        reward += delivered_this_step * self.DELIVERY_BONUS
        self.deliveries += delivered_this_step

        # -- Waiting penalty (scaled) --
        total_waiting = self.building.total_waiting_passengers()
        reward += self.WAIT_PENALTY * total_waiting

        # -- Advance building --
        self.building.step()

        self.total_reward += reward
        done = self.step_count >= self.episode_length

        info = {
            "step":        self.step_count,
            "floor":       self.current_floor,
            "direction":   self.direction,
            "waiting":     total_waiting,
            "in_elevator": len(self.passengers_in_elevator),
            "deliveries":  self.deliveries,
            "energy":      self.energy_used,
            "avg_wait":    self.building.average_waiting_time(),
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------

    def _get_state(self) -> Tuple[int, int, int, int]:
        """
        Compact state: (floor, dir_idx, nearest_req, load_bucket)
        """
        floor = self.current_floor
        dir_idx = self.direction + 1  # {-1,0,1} -> {0,1,2}

        # Nearest pending request floor
        pending = self.building.get_pending_requests()
        # Also consider destinations of passengers in elevator
        dests = [p.destination_floor for p in self.passengers_in_elevator]
        all_targets = pending + dests
        if all_targets:
            nearest = min(all_targets, key=lambda f: abs(f - floor))
        else:
            nearest = self.num_floors  # sentinel: no request

        # Load bucket: 0=empty, 1=light(1-3), 2=heavy(4+)
        n_in = len(self.passengers_in_elevator)
        if n_in == 0:
            load = 0
        elif n_in <= 3:
            load = 1
        else:
            load = 2

        return (floor, dir_idx, nearest, load)

    def state_to_index(self, state: Tuple[int, int, int, int]) -> int:
        """Flatten compact state to single integer index."""
        floor, dir_idx, nearest, load = state
        return (
            floor * self.n_directions * self.n_req_states * self.n_load_states
            + dir_idx * self.n_req_states * self.n_load_states
            + nearest * self.n_load_states
            + load
        )

    @property
    def state_space_size(self) -> int:
        return (
            self.n_floors
            * self.n_directions
            * self.n_req_states
            * self.n_load_states
        )

    # ------------------------------------------------------------------

    def _pickup_passengers(self) -> int:
        floor = self.building.floors[self.current_floor]
        cap_left = self.max_capacity - len(self.passengers_in_elevator)
        to_board = list(floor.waiting_passengers[:cap_left])
        for p in to_board:
            p.pickup_time = self.building.current_step
            floor.remove_passenger(p)
            self.passengers_in_elevator.append(p)
        return len(to_board)

    def _dropoff_passengers(self) -> int:
        to_deliver = [p for p in self.passengers_in_elevator
                      if p.destination_floor == self.current_floor]
        for p in to_deliver:
            self.building.deliver_passenger(p, self.building.current_step)
            self.passengers_in_elevator.remove(p)
        return len(to_deliver)

    def _count_new_deliveries(self) -> int:
        return sum(1 for p in self.building.delivered_passengers
                   if p.delivery_time == self.building.current_step)

    # ------------------------------------------------------------------

    def render(self) -> str:
        """Plain ASCII render (no Unicode, no emoji) — safe on Windows."""
        lines = []
        lines.append(f"Step {self.step_count} | Floor {self.current_floor} | "
                      f"Dir: {self._dir_str()} | InElev: {len(self.passengers_in_elevator)}")
        lines.append("=" * 52)
        for fid in range(self.num_floors - 1, -1, -1):
            fl = self.building.floors[fid]
            elev = "[ELEV]" if fid == self.current_floor else "      "
            bar = "#" * min(fl.queue_length(), 10)
            lines.append(f"  F{fid:2d} {elev}  [{bar:<10}] {fl.queue_length():>2} waiting")
        lines.append("=" * 52)
        lines.append(f"Reward: {self.total_reward:.1f} | "
                     f"Delivered: {self.deliveries} | Energy: {self.energy_used}")
        return "\n".join(lines)

    def _dir_str(self) -> str:
        return {DIR_UP: "UP", DIR_DOWN: "DOWN", DIR_IDLE: "IDLE"}[self.direction]

    def get_episode_stats(self) -> Dict[str, Any]:
        return {
            "total_reward":   round(self.total_reward, 2),
            "deliveries":     self.deliveries,
            "energy_used":    self.energy_used,
            "avg_wait_time":  round(self.building.average_waiting_time(), 2),
            "avg_total_time": round(self.building.average_total_time(), 2),
            "total_generated": len(self.building.all_passengers),
            "still_waiting":  self.building.total_waiting_passengers(),
        }
