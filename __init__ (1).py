"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Deep Q-Network (DQN) Agent for RPOD                                        ║
║  dqn_agent.py                                                                ║
║                                                                              ║
║  Architecture:                                                               ║
║    • Double DQN  (target network + online network)                           ║
║    • Prioritized Experience Replay (PER)                                     ║
║    • Dueling network head (value + advantage streams)                        ║
║    • ε-greedy exploration with linear decay                                  ║
║    • Discrete action space (27 thrust combinations)                          ║
║                                                                              ║
║  Action discretization:                                                      ║
║    Each axis: {-max_thrust, 0, +max_thrust}  →  3³ = 27 actions             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, Optional
import random
import logging
import os

logger = logging.getLogger(__name__)

# ── Experience tuple ───────────────────────────────────────────────────────
Transition = namedtuple("Transition",
    ["state", "action", "reward", "next_state", "done"])


# ── Dueling DQN Network ───────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling architecture separates value V(s) and advantage A(s,a),
    leading to faster convergence in continuous-control tasks.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: Tuple[int, ...] = (256, 256, 128)):
        super().__init__()
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes[:-1]:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.features = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1),
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.features(x)
        val   = self.value_stream(feat)
        adv   = self.adv_stream(feat)
        q_val = val + adv - adv.mean(dim=1, keepdim=True)
        return q_val


# ── Prioritized Replay Buffer ─────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Sum-tree based PER with importance-sampling weights.
    Prioritizes transitions with high TD-error.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame_idx    = 0
        self.buffer       = deque(maxlen=capacity)
        self.priorities   = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(self, *args):
        self.buffer.append(Transition(*args))
        self.priorities.append(self._max_priority)

    def sample(self, batch_size: int) -> Tuple:
        n       = len(self.buffer)
        probs   = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs  /= probs.sum()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        beta = min(1.0, self.beta_start +
                   self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_idx += 1

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch])
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        for idx, err in zip(indices, errors):
            p = (abs(float(err)) + 1e-6)
            self.priorities[idx] = p
            self._max_priority = max(self._max_priority, p)

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Double DQN + Dueling + PER agent for RPOD discrete control.

    Action encoding:
        27 actions = 3 thrust levels per axis (−T, 0, +T)
        index → [ax_idx, ay_idx, az_idx] → actual thrust values
    """

    def __init__(self,
                 obs_dim:          int   = 8,
                 max_thrust:       float = 0.004,
                 lr:               float = 3e-4,
                 gamma:            float = 0.995,
                 eps_start:        float = 1.0,
                 eps_end:          float = 0.05,
                 eps_decay_steps:  int   = 80_000,
                 batch_size:       int   = 256,
                 target_update:    int   = 500,
                 buffer_capacity:  int   = 100_000,
                 hidden_sizes:     Tuple = (256, 256, 128),
                 device:           str   = "auto"):

        self.obs_dim    = obs_dim
        self.max_thrust = max_thrust
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"DQN agent using device: {self.device}")

        # Action space: 3 levels × 3 axes = 27 discrete actions
        self.n_actions  = 27
        self._thrust_levels = np.array([-max_thrust, 0.0, max_thrust])
        self._action_map = np.array(
            [[i, j, k] for i in range(3) for j in range(3) for k in range(3)]
        )  # shape (27, 3)

        # Networks
        self.online_net = DuelingDQN(obs_dim, self.n_actions, hidden_sizes).to(self.device)
        self.target_net = DuelingDQN(obs_dim, self.n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=eps_decay_steps, eta_min=lr * 0.1)

        # Replay
        self.memory = PrioritizedReplayBuffer(buffer_capacity)

        # Epsilon
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = (eps_start - eps_end) / eps_decay_steps
        self.step_count   = 0
        self.episode_count = 0

        # Stats
        self.loss_history    = []
        self.reward_history  = []
        self.q_val_history   = []

    # ──────────────────────────────────────────────────────────────────────
    def decode_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous thrust vector."""
        idx = self._action_map[action_idx]
        return self._thrust_levels[idx].copy()

    def encode_action(self, thrust: np.ndarray) -> int:
        """Find closest discrete action to a continuous thrust vector."""
        diffs = np.linalg.norm(
            self._thrust_levels[self._action_map] - thrust, axis=1)
        return int(np.argmin(diffs))

    # ──────────────────────────────────────────────────────────────────────
    def select_action(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """ε-greedy action selection; returns continuous thrust."""
        if training and random.random() < self.eps:
            idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                t   = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q   = self.online_net(t)
                idx = int(q.argmax(1).item())
                if training:
                    self.q_val_history.append(float(q.max().item()))

        if training:
            self.step_count += 1
            self.eps = max(self.eps_end, self.eps - self.eps_decay)

        return self.decode_action(idx), idx

    # ──────────────────────────────────────────────────────────────────────
    def store_transition(self, state, action_idx, reward, next_state, done):
        self.memory.push(
            np.array(state,      dtype=np.float32),
            action_idx,
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )

    # ──────────────────────────────────────────────────────────────────────
    def update(self) -> Optional[float]:
        """Sample a minibatch and perform one gradient step. Returns loss."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        weights     = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        q_vals = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: use online net for action selection, target net for evaluation
        with torch.no_grad():
            best_actions  = self.online_net(next_states).argmax(1)
            target_q_next = self.target_net(next_states).gather(
                1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * target_q_next * (1 - dones)

        # Weighted Huber loss (robust to outliers)
        td_errors = (q_vals - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(q_vals, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update PER priorities
        self.memory.update_priorities(indices, np.abs(td_errors))

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        # Periodically sync target network
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_val

    # ──────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "step_count":         self.step_count,
            "episode_count":      self.episode_count,
            "eps":                self.eps,
            "loss_history":       self.loss_history[-1000:],
            "reward_history":     self.reward_history[-1000:],
        }, path)
        logger.info(f"DQN model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step_count    = ckpt.get("step_count", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.eps           = ckpt.get("eps", self.eps_end)
        logger.info(f"DQN model loaded ← {path}")

    def get_stats(self) -> dict:
        return {
            "step_count":    self.step_count,
            "epsilon":       round(self.eps, 4),
            "buffer_size":   len(self.memory),
            "mean_loss":     float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
            "mean_q":        float(np.mean(self.q_val_history[-100:])) if self.q_val_history else 0.0,
            "mean_reward":   float(np.mean(self.reward_history[-100:])) if self.reward_history else 0.0,
        }
