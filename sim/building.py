"""
building.py — Passenger, Floor, and Building simulation models
for the Smart Elevator Scheduling System.

SDG 11 – Sustainable Cities and Communities
SDG 9  – Industry, Innovation and Infrastructure
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Passenger:
    """Represents a single passenger waiting for or riding the elevator."""
    passenger_id: int
    origin_floor: int
    destination_floor: int
    arrival_time: int          # simulation step when they arrived
    pickup_time: Optional[int] = None   # step when they boarded
    delivery_time: Optional[int] = None  # step when they exited

    @property
    def wait_time(self) -> Optional[int]:
        """Steps waited before boarding."""
        if self.pickup_time is None:
            return None
        return self.pickup_time - self.arrival_time

    @property
    def travel_time(self) -> Optional[int]:
        """Steps spent inside elevator."""
        if self.delivery_time is None or self.pickup_time is None:
            return None
        return self.delivery_time - self.pickup_time

    @property
    def total_time(self) -> Optional[int]:
        """Total steps from arrival to delivery."""
        if self.delivery_time is None:
            return None
        return self.delivery_time - self.arrival_time

    @property
    def direction(self) -> int:
        """+1 = going up, -1 = going down."""
        return 1 if self.destination_floor > self.origin_floor else -1


@dataclass
class Floor:
    """Represents one floor in the building."""
    floor_id: int
    waiting_passengers: List[Passenger] = field(default_factory=list)

    def add_passenger(self, passenger: Passenger):
        self.waiting_passengers.append(passenger)

    def remove_passenger(self, passenger: Passenger):
        self.waiting_passengers.remove(passenger)

    def has_request(self) -> bool:
        return len(self.waiting_passengers) > 0

    def queue_length(self) -> int:
        return len(self.waiting_passengers)


class Building:
    """
    Simulates a building with N floors and a passenger arrival process.

    Passengers arrive randomly on floors and request to go to other floors.
    Arrival rate is configurable (Poisson process).
    """

    def __init__(
        self,
        num_floors: int = 10,
        arrival_rate: float = 0.3,    # avg passengers arriving per step
        seed: Optional[int] = 42,
        peak_floors: Optional[List[int]] = None,  # high-demand floors
    ):
        self.num_floors = num_floors
        self.arrival_rate = arrival_rate
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.peak_floors = peak_floors or [0, num_floors - 1]  # ground + top

        self.floors: List[Floor] = [Floor(i) for i in range(num_floors)]
        self._passenger_counter = 0
        self.current_step = 0

        # Statistics
        self.delivered_passengers: List[Passenger] = []
        self.all_passengers: List[Passenger] = []

    def reset(self):
        """Reset building state for a new episode."""
        self.floors = [Floor(i) for i in range(self.num_floors)]
        self._passenger_counter = 0
        self.current_step = 0
        self.delivered_passengers = []
        self.all_passengers = []

    def step(self):
        """
        Advance one simulation step.
        Generates new passenger arrivals using Poisson process.
        """
        self.current_step += 1
        new_passengers = []

        # Poisson arrivals — each floor can get a new passenger
        for floor_id in range(self.num_floors):
            # Increase rate for peak floors
            rate = self.arrival_rate
            if floor_id in self.peak_floors:
                rate *= 2.0

            if self.np_rng.poisson(rate) > 0:
                # Generate destination (different from origin)
                possible_dests = [f for f in range(self.num_floors) if f != floor_id]
                dest = self.rng.choice(possible_dests)

                passenger = Passenger(
                    passenger_id=self._passenger_counter,
                    origin_floor=floor_id,
                    destination_floor=dest,
                    arrival_time=self.current_step,
                )
                self._passenger_counter += 1
                self.floors[floor_id].add_passenger(passenger)
                self.all_passengers.append(passenger)
                new_passengers.append(passenger)

        return new_passengers

    def get_pending_requests(self) -> List[int]:
        """Returns list of floor IDs with waiting passengers."""
        return [f.floor_id for f in self.floors if f.has_request()]

    def get_request_bitmask(self) -> int:
        """Bitmask of floors with pending requests (bit i = floor i has request)."""
        mask = 0
        for f in self.floors:
            if f.has_request():
                mask |= (1 << f.floor_id)
        return mask

    def get_queue_lengths(self) -> List[int]:
        """Queue length on each floor."""
        return [f.queue_length() for f in self.floors]

    def get_passengers_on_floor(self, floor_id: int) -> List[Passenger]:
        """Get waiting passengers on a specific floor."""
        return list(self.floors[floor_id].waiting_passengers)

    def deliver_passenger(self, passenger: Passenger, current_step: int):
        """Mark a passenger as delivered."""
        passenger.delivery_time = current_step
        self.delivered_passengers.append(passenger)

    # ── Statistics ──────────────────────────────────────────────────────────

    def average_waiting_time(self) -> float:
        """Average waiting time for all passengers picked up so far."""
        wait_times = [p.wait_time for p in self.all_passengers if p.wait_time is not None]
        return float(np.mean(wait_times)) if wait_times else 0.0

    def average_total_time(self) -> float:
        """Average total time (wait + travel) for delivered passengers."""
        total_times = [p.total_time for p in self.delivered_passengers if p.total_time is not None]
        return float(np.mean(total_times)) if total_times else 0.0

    def total_waiting_passengers(self) -> int:
        return sum(f.queue_length() for f in self.floors)

    def summary(self) -> dict:
        return {
            "step": self.current_step,
            "total_passengers_generated": len(self.all_passengers),
            "total_delivered": len(self.delivered_passengers),
            "still_waiting": self.total_waiting_passengers(),
            "avg_wait_time": round(self.average_waiting_time(), 2),
            "avg_total_time": round(self.average_total_time(), 2),
            "queue_lengths": self.get_queue_lengths(),
        }
