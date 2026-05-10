"""
visualizer.py -- ASCII Terminal Visualizer for Elevator Simulation.
No Unicode/emoji -- safe on all Windows terminals.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sim.elevator_env import ElevatorEnv, Action
from sim.building import Building


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render_frame(env: ElevatorEnv, step: int, action_name: str,
                 episode_reward: float, policy_name: str = "RL Policy") -> str:
    lines = []
    building = env.building

    lines.append("=" * 62)
    lines.append(f"  Smart Elevator RL Simulator  --  {policy_name}")
    lines.append(f"  SDG 11: Sustainable Cities  |  SDG 9: Innovation")
    lines.append("=" * 62)
    lines.append(f"  Step: {step:<5}  Action: {action_name:<12}  Reward: {episode_reward:.1f}")
    lines.append("")

    for floor_id in range(env.num_floors - 1, -1, -1):
        fl = building.floors[floor_id]
        q  = fl.queue_length()

        if floor_id == env.current_floor:
            dir_char = {"UP": "^", "DOWN": "v", "IDLE": "="}[env._dir_str()]
            elev = f"[{dir_char}ELEV{dir_char}]"
        else:
            elev = "         "

        bar = "#" * min(q, 14) + "." * (14 - min(q, 14))

        dests = [p for p in env.passengers_in_elevator
                 if p.destination_floor == floor_id]
        dest_mark = " <-- DEST" if dests else ""

        lines.append(f"  F{floor_id:02d} | {elev} | [{bar}] {q:2d} waiting{dest_mark}")

    lines.append("")
    lines.append("-" * 62)
    in_elev   = len(env.passengers_in_elevator)
    delivered = env.deliveries
    energy    = env.energy_used
    avg_wait  = building.average_waiting_time()
    total_wait = building.total_waiting_passengers()

    lines.append(f"  In elevator : {in_elev}/{env.max_capacity}"
                 f"   |   Delivered: {delivered}"
                 f"   |   Energy: {energy}")
    lines.append(f"  Avg Wait    : {avg_wait:.1f} steps"
                 f"   |   Total Waiting: {total_wait}")
    lines.append("=" * 62)
    return "\n".join(lines)


def animate_episode(env: ElevatorEnv, policy_fn, policy_name="Policy",
                    delay=0.08, max_steps=200):
    """Run one animated episode. policy_fn(env) -> action int."""
    state = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    action_name = "START"

    while not done and step < max_steps:
        clear()
        print(render_frame(env, step, action_name, total_reward, policy_name))
        time.sleep(delay)

        action = policy_fn(env)
        action_name = ["MOVE_UP", "MOVE_DOWN", "STAY", "OPEN_DOOR"][action]
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    clear()
    print(render_frame(env, step, "DONE", total_reward, policy_name))
    stats = env.get_episode_stats()
    print(f"\n  Episode complete!")
    print(f"  Total Reward : {stats['total_reward']:.2f}")
    print(f"  Deliveries   : {stats['deliveries']}")
    print(f"  Avg Wait     : {stats['avg_wait_time']:.2f} steps")
    print(f"  Energy Used  : {stats['energy_used']} moves")
    return stats
