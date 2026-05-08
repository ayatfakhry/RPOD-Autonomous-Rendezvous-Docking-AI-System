"""
evaluator.py — Performance Evaluation & Benchmarking
=====================================================
Evaluates trained AI agents and compares against the
classical PD controller (GNSS-IMU baseline).

Metrics computed per evaluation run:
  • Mean / std episode reward
  • Docking success rate
  • Mean final range (m)
  • Mean fuel consumed (kg)
  • Mean time-to-dock (s)
  • Mean relative velocity at docking (m/s)
  • RMSE of trajectory vs ideal approach
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ClassicalBaseline:
    """
    Classical PD Controller (GNSS-IMU baseline).

    Implements a proportional-derivative thrust law:
      a = −Kp·r − Kd·v

    Gains are scheduled by mission phase for realistic comparison:
      Far field (>500m) : gentle correction
      Mid range         : moderate gains
      Proximity (<30m)  : fine control
    """

    def __init__(self, max_thrust: float = 0.003):
        self.max_thrust = max_thrust

    def act(self, obs: np.ndarray, deterministic: bool = True):
        """Extract state from observation and apply PD control."""
        # obs[:6] = [px,py,pz,vx,vy,vz] / normalisation_scales
        pos = obs[:3] * 2500.0
        vel = obs[3:6] * 2.0
        dist = float(np.linalg.norm(pos))

        # Phase-scheduled gains
        if dist > 500:
            Kp, Kd = 0.0003, 0.015
        elif dist > 150:
            Kp, Kd = 0.0004, 0.018
        elif dist > 30:
            Kp, Kd = 0.0005, 0.022
        elif dist > 5:
            Kp, Kd = 0.0008, 0.030
        else:
            Kp, Kd = 0.0015, 0.050

        ax = -Kp * pos[0] - Kd * vel[0]
        ay = -Kp * pos[1] - Kd * vel[1]
        az = -Kp * pos[2] - Kd * vel[2]

        action = np.clip([ax, ay, az], -self.max_thrust, self.max_thrust)
        return action, -1   # (-1 = no discrete action index)

    def end_episode(self):
        pass


class Evaluator:
    """
    Evaluation harness for RPOD agents.

    Supports:
      - "ppo"       : PPOAgent (continuous actions via act())
      - "dqn"       : DQNAgent (discrete+continuous via act())
      - "classical" : ClassicalBaseline

    Usage:
        evaluator = Evaluator(env, agent, mode="ppo")
        stats = evaluator.run(n_episodes=20, deterministic=True)
        print(stats)
    """

    def __init__(self, env, agent, mode: str = "ppo"):
        self.env   = env
        self.agent = agent
        self.mode  = mode

    def run(
        self,
        n_episodes: int = 20,
        deterministic: bool = True,
        seed_offset: int = 9999,
    ) -> Dict[str, float]:
        """Run n_episodes and return aggregated metrics."""
        rewards       = []
        final_ranges  = []
        min_ranges    = []
        fuel_useds    = []
        step_counts   = []
        dock_flags    = []
        final_speeds  = []
        episode_logs: List[List[Dict]] = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed_offset + ep)
            ep_reward = 0.0
            done      = False
            ep_log: List[Dict] = []

            while not done:
                # Get action from agent
                if self.mode == "ppo":
                    action, _, _ = self.agent.act(obs, deterministic=deterministic)
                elif self.mode == "dqn":
                    action, _ = self.agent.act(obs, deterministic=deterministic)
                elif self.mode == "classical":
                    action, _ = self.agent.act(obs, deterministic=True)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                obs = next_obs

                ep_log.append({
                    "dist":  info.get("dist", 0),
                    "speed": info.get("speed", 0),
                    "fuel":  info.get("fuel", 0),
                })

            stats = self.env.get_episode_stats()
            rewards.append(ep_reward)
            final_ranges.append(stats.get("ep_final_range", 0))
            min_ranges.append(stats.get("ep_min_range", 0))
            fuel_useds.append(stats.get("ep_fuel_used", 0))
            step_counts.append(stats.get("ep_steps", 0))
            dock_flags.append(stats.get("ep_docked", 0))
            final_speeds.append(ep_log[-1]["speed"] if ep_log else 0.0)
            episode_logs.append(ep_log)

        # Aggregate
        result = {
            "mean_reward":      float(np.mean(rewards)),
            "std_reward":       float(np.std(rewards)),
            "min_reward":       float(np.min(rewards)),
            "max_reward":       float(np.max(rewards)),
            "dock_rate":        float(np.mean(dock_flags)),
            "mean_final_range": float(np.mean(final_ranges)),
            "std_final_range":  float(np.std(final_ranges)),
            "mean_min_range":   float(np.mean(min_ranges)),
            "mean_fuel_used":   float(np.mean(fuel_useds)),
            "std_fuel_used":    float(np.std(fuel_useds)),
            "mean_steps":       float(np.mean(step_counts)),
            "mean_final_speed": float(np.mean(final_speeds)),
            "n_episodes":       n_episodes,
            "mode":             self.mode,
        }

        logger.debug(
            f"[{self.mode.upper()}] "
            f"Reward={result['mean_reward']:.1f}±{result['std_reward']:.1f} "
            f"Dock={result['dock_rate']*100:.1f}% "
            f"Range={result['mean_final_range']:.2f}m"
        )
        return result

    def generate_report(self, stats: Dict[str, Any]) -> str:
        """Format a human-readable evaluation report."""
        lines = [
            "=" * 60,
            f"  RPOD EVALUATION REPORT — {stats['mode'].upper()}",
            "=" * 60,
            f"  Episodes evaluated : {stats['n_episodes']}",
            f"  Mean reward        : {stats['mean_reward']:+.2f} ± {stats['std_reward']:.2f}",
            f"  Docking success    : {stats['dock_rate']*100:.1f}%",
            f"  Mean final range   : {stats['mean_final_range']:.4f} m",
            f"  Min range achieved : {stats['mean_min_range']:.4f} m",
            f"  Mean fuel used     : {stats['mean_fuel_used']:.3f} kg",
            f"  Mean steps/episode : {stats['mean_steps']:.0f}",
            f"  Final speed (mean) : {stats['mean_final_speed']:.4f} m/s",
            "=" * 60,
        ]
        return "\n".join(lines)
