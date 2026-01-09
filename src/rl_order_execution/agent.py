import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque
from typing import Tuple, Deque, Optional, cast
from gymnasium import spaces

from rl_order_execution.settings import Settings

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.device = device

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_space: spaces.Space,
        settings: Settings,
        device: torch.device,
    ):
        """
        Initializes the DQN Agent.

        Args:
            state_dim: Dimension of the state space.
            action_space: The Gym action space (Must be Discrete).
            settings: Configuration settings.
            device: Torch device.
        """
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = int(action_space.n)
        else:
            raise ValueError(
                f"DQNAgent requires a Discrete action space, got {type(action_space)}"
            )

        self.state_dim = state_dim
        self.settings = settings
        self.device = device
        self.memory = ReplayBuffer(settings.rl.memory_size, device)
        self.epsilon = settings.rl.epsilon_start

        logger.info(
            f"Initialized DQNAgent with state_dim={state_dim}, action_dim={self.action_dim} on {device}"
        )

        self.policy_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net = DQN(state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=settings.rl.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray, is_eval: bool = False) -> int:
        if not is_eval and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.settings.rl.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.settings.rl.batch_size
        )

        curr_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.settings.rl.gamma * next_q * (1 - dones))

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_epsilon(self) -> None:
        self.epsilon = max(
            self.settings.rl.epsilon_end, self.epsilon * self.settings.rl.epsilon_decay
        )

    def update_target_network(self) -> None:
        logger.info("Updating target network weights")
        self.target_net.load_state_dict(self.policy_net.state_dict())
