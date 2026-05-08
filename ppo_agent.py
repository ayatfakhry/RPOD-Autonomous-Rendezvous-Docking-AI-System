"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RPOD Autonomous System — Gymnasium-Compatible RL Environment               ║
║  rpod_env.py                                                                 ║
║                                                                              ║
║  Physics: Clohessy-Wiltshire (CW) relative orbital mechanics                ║
║  State:   [x, y, z, vx, vy, vz, fuel_remaining, time_step]  (8-dim)        ║
║  Action:  [ax, ay, az]  continuous thrust in m/s²  (3-dim)                  ║
║                                                                              ║
║  Reference orbit: ISS LEO ~408 km altitude                                  ║
║  Author:  RPOD AI Project                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ── Physical constants ─────────────────────────────────────────────────────
MU          = 3.986004418e14   # m³/s²  Earth gravitational parameter
R_EARTH     = 6.3781e6         # m      Earth equatorial radius
ALT_ISS     = 408e3            # m      ISS orbital altitude
R_ORBIT     = R_EARTH + ALT_ISS
N_MOTION    = np.sqrt(MU / R_ORBIT**3)  # rad/s  mean motion ≈ 0.001128


@dataclass
class EnvConfig:
    """Simulation and reward configuration for the RPOD environment."""
    # Initial state bounds
    init_range_min:     float = 200.0     # m
    init_range_max:     float = 2500.0    # m
    init_vel_noise:     float = 0.3       # m/s std-dev
    init_z_max:         float = 100.0     # m

    # RCS limits
    max_thrust:         float = 0.004     # m/s²  (reaction control system)
    max_fuel:           float = 100.0     # arbitrary units

    # Episode termination
    dock_radius:        float = 0.3       # m  — success
    dock_speed_max:     float = 0.05      # m/s  — safe docking speed
    max_range:          float = 5000.0    # m  — out-of-range abort
    max_steps:          int   = 4000      # steps per episode
    dt:                 float = 0.5       # s  integration timestep

    # Reward shaping weights
    w_distance:         float = -0.015
    w_velocity:         float = -0.008
    w_fuel:             float = -0.002
    w_align:            float = 0.005     # reward for pointing toward target
    r_dock_success:     float = 500.0
    r_dock_unsafe:      float = -150.0   # docked but too fast
    r_oob:              float = -200.0   # out of bounds
    r_fuel_empty:       float = -100.0

    # Fault injection (for robustness training)
    gnss_fault_prob:    float = 0.01     # per-step GNSS degradation
    gnss_noise_normal:  float = 0.5      # m
    gnss_noise_fault:   float = 8.0      # m
    debris_enabled:     bool  = True
    num_debris:         int   = 8


@dataclass
class EpisodeStats:
    """Collects per-episode metrics for evaluation."""
    total_reward:    float = 0.0
    total_fuel_used: float = 0.0
    steps:           int   = 0
    min_distance:    float = np.inf
    final_distance:  float = 0.0
    success:         bool  = False
    collision:       bool  = False
    fault_steps:     int   = 0
    trajectory:      list  = field(default_factory=list)


class RPODEnvironment(gym.Env):
    """
    Gymnasium environment for Autonomous Rendezvous, Proximity Operations,
    and Docking (RPOD) using Clohessy-Wiltshire relative orbital mechanics.

    Observation space (8-dim, normalized):
        [x, y, z, vx, vy, vz, fuel_fraction, progress]

    Action space (3-dim, continuous):
        [ax, ay, az]  thrust in [-max_thrust, +max_thrust] m/s²

    Reward function:
        r = w_dist * dist + w_vel * speed + w_fuel * |a|
            + w_align * cos(heading_error)
            + terminal_bonus (dock / OOB / fuel)

    Phases (auto-detected):
        0: Far Field     (> 500 m)
        1: Mid Range     (200–500 m)
        2: Proximity Ops (30–200 m)
        3: Docking       (< 30 m)
        4: Final         (< 5 m)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self._rng = np.random.default_rng()

        # ── Observation space (normalized to [-1, 1]) ──────────────────────
        obs_high = np.array([1.0] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # ── Action space (continuous thrust) ──────────────────────────────
        act_high = np.full(3, self.cfg.max_thrust, dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)

        # ── Normalization constants ────────────────────────────────────────
        self._pos_norm  = self.cfg.max_range
        self._vel_norm  = 2.0        # m/s
        self._fuel_norm = self.cfg.max_fuel

        # ── Internal state ─────────────────────────────────────────────────
        self._state     = np.zeros(6)   # [x,y,z,vx,vy,vz]
        self._fuel      = self.cfg.max_fuel
        self._step_count = 0
        self._stats     = EpisodeStats()
        self._debris     = np.zeros((self.cfg.num_debris, 6))  # [x,y,z,vx,vy,vz]
        self._gnss_fault = False
        self._phase      = 0
        self._episode    = 0

    # ──────────────────────────────────────────────────────────────────────
    # GYMNASIUM API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize initial chaser position in CW frame
        dist  = self._rng.uniform(self.cfg.init_range_min, self.cfg.init_range_max)
        theta = self._rng.uniform(0, 2 * np.pi)
        phi   = self._rng.uniform(-np.pi / 6, np.pi / 6)

        x  = dist * np.cos(theta) * np.cos(phi)
        y  = dist * np.sin(theta) * np.cos(phi)
        z  = dist * np.sin(phi) * (self.cfg.init_z_max / self.cfg.init_range_max)
        vx = self._rng.normal(-0.5, self.cfg.init_vel_noise)
        vy = self._rng.normal(0.0,  self.cfg.init_vel_noise)
        vz = self._rng.normal(0.0,  self.cfg.init_vel_noise * 0.3)

        self._state     = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
        self._fuel      = self.cfg.max_fuel
        self._step_count = 0
        self._gnss_fault = False
        self._phase      = 0
        self._episode   += 1
        self._stats      = EpisodeStats()

        if self.cfg.debris_enabled:
            self._init_debris()

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -self.cfg.max_thrust, self.cfg.max_thrust)

        # ── Fault injection ───────────────────────────────────────────────
        self._gnss_fault = self._rng.random() < self.cfg.gnss_fault_prob
        if self._gnss_fault:
            self._stats.fault_steps += 1

        # ── Propagate state (CW equations) ────────────────────────────────
        self._state = self._cw_step(self._state, action, self.cfg.dt)

        # ── Update debris ─────────────────────────────────────────────────
        if self.cfg.debris_enabled:
            self._debris[:, :3] += self._debris[:, 3:] * self.cfg.dt

        # ── Fuel cost ─────────────────────────────────────────────────────
        thrust_mag  = float(np.linalg.norm(action))
        fuel_cost   = thrust_mag * self.cfg.dt * 180.0
        self._fuel -= fuel_cost
        self._fuel  = max(0.0, self._fuel)

        # ── Metrics ───────────────────────────────────────────────────────
        dist  = float(np.linalg.norm(self._state[:3]))
        speed = float(np.linalg.norm(self._state[3:6]))
        self._step_count  += 1
        self._stats.steps  = self._step_count
        self._stats.min_distance = min(self._stats.min_distance, dist)
        self._stats.total_fuel_used += fuel_cost

        # Update phase
        self._phase = self._get_phase(dist)

        # Record trajectory (sparse)
        if self._step_count % 10 == 0:
            self._stats.trajectory.append({
                "t": self._step_count * self.cfg.dt,
                "dist": dist,
                "speed": speed,
                "fuel": self._fuel,
            })

        # ── Reward ────────────────────────────────────────────────────────
        reward, terminated, truncated = self._compute_reward(dist, speed, thrust_mag)
        self._stats.total_reward += reward
        self._stats.final_distance = dist

        obs  = self._get_obs()
        info = self._get_info()
        info["phase"]       = self._phase
        info["fuel"]        = self._fuel
        info["dist"]        = dist
        info["speed"]       = speed
        info["gnss_fault"]  = self._gnss_fault
        info["stats"]       = self._stats if terminated or truncated else None

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # PHYSICS
    # ──────────────────────────────────────────────────────────────────────

    def _cw_step(self, s: np.ndarray, a: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate one timestep using analytical Clohessy-Wiltshire solution.
        State vector: [x, y, z, vx, vy, vz]
        """
        n = N_MOTION
        x, y, z, vx, vy, vz = s
        ax, ay, az = a
        c, ss = np.cos(n * dt), np.sin(n * dt)

        nx  = (4 - 3*c)*x + ss/n*vx + 2*(1 - c)/n*vy + 0.5*ax*dt**2
        ny  = 6*(ss - n*dt)*x + y - 2*(c - 1)/n*vx + (4*ss/n - 3*dt)*vy + 0.5*ay*dt**2
        nz  = z*c + vz/n*ss + 0.5*az*dt**2
        nvx = 3*n*ss*x + c*vx + 2*ss*vy + ax*dt
        nvy = -6*n*(1 - c)*x - 2*ss*vx + (4*c - 3)*vy + ay*dt
        nvz = -n*z*ss + vz*c + az*dt
        return np.array([nx, ny, nz, nvx, nvy, nvz])

    # ──────────────────────────────────────────────────────────────────────
    # REWARD
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(self, dist: float, speed: float, thrust_mag: float
                        ) -> Tuple[float, bool, bool]:
        cfg = self.cfg
        terminated = truncated = False

        # Shaping terms
        r_dist  = cfg.w_distance * dist
        r_speed = cfg.w_velocity * speed
        r_fuel  = cfg.w_fuel     * thrust_mag

        # Alignment bonus: reward heading toward origin
        pos  = self._state[:3]
        vel  = self._state[3:6]
        d    = np.linalg.norm(pos)
        v    = np.linalg.norm(vel)
        cos_align = 0.0
        if d > 0.1 and v > 0.001:
            cos_align = float(-np.dot(pos / d, vel / v))  # positive when closing
        r_align = cfg.w_align * max(0.0, cos_align)

        reward = r_dist + r_speed + r_fuel + r_align

        # ── Terminal conditions ────────────────────────────────────────────
        if dist < cfg.dock_radius:
            if speed < cfg.dock_speed_max:
                reward += cfg.r_dock_success
                terminated = True
                self._stats.success = True
                logger.debug(f"[Ep {self._episode}] DOCKED at step {self._step_count}, dist={dist:.3f}m")
            else:
                reward += cfg.r_dock_unsafe
                terminated = True
                logger.debug(f"[Ep {self._episode}] UNSAFE DOCK speed={speed:.3f}m/s")

        elif dist > cfg.max_range:
            reward += cfg.r_oob
            terminated = True

        elif self._fuel <= 0:
            reward += cfg.r_fuel_empty
            terminated = True

        elif self._step_count >= cfg.max_steps:
            truncated = True

        return reward, terminated, truncated

    # ──────────────────────────────────────────────────────────────────────
    # OBSERVATIONS
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return normalized observation vector."""
        # Optionally add GNSS noise
        noise = self.cfg.gnss_noise_fault if self._gnss_fault else self.cfg.gnss_noise_normal
        pos_noisy = self._state[:3] + self._rng.normal(0, noise, 3)
        vel_exact = self._state[3:6]  # IMU-derived, lower noise

        obs = np.concatenate([
            pos_noisy  / self._pos_norm,
            vel_exact  / self._vel_norm,
            [self._fuel / self._fuel_norm],
            [self._step_count / self.cfg.max_steps],
        ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step":    self._step_count,
            "episode": self._episode,
            "phase":   self._phase,
        }

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_phase(self, dist: float) -> int:
        if dist > 500:   return 0
        if dist > 200:   return 1
        if dist > 30:    return 2
        if dist > 5:     return 3
        return 4

    def _init_debris(self):
        """Scatter debris in the neighborhood."""
        pos = self._rng.uniform(-1500, 1500, (self.cfg.num_debris, 3))
        vel = self._rng.uniform(-0.2, 0.2, (self.cfg.num_debris, 3))
        self._debris = np.hstack([pos, vel]).astype(np.float64)

    def get_phase_name(self) -> str:
        names = ["FAR_FIELD", "MID_RANGE", "PROXIMITY_OPS", "DOCKING", "FINAL"]
        return names[self._phase]

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def fuel(self) -> float:
        return self._fuel

    @property
    def phase(self) -> int:
        return self._phase
