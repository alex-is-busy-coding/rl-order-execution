import pytest
import numpy as np
import torch

from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent
from rl_order_execution.settings import get_settings


@pytest.fixture
def env():
    settings = get_settings()
    return OrderExecutionEnv(settings)


@pytest.fixture
def agent(env):
    settings = get_settings()
    device = torch.device("cpu")
    return DQNAgent(
        state_dim=3, action_space=env.action_space, settings=settings, device=device
    )


def test_environment_initialization(env):
    """Test that the environment resets correctly."""
    state, info = env.reset()

    assert len(state) == 3
    assert env.shares_remaining == env.total_shares
    assert env.time_remaining == env.time_horizon
    assert 0 <= state[0] <= 1.0
    assert 0 <= state[1] <= 1.0


def test_step_mechanics(env):
    """Test that stepping through the environment updates state correctly."""
    env.reset()
    initial_shares = env.shares_remaining
    initial_time = env.time_remaining

    action_idx = 2
    next_state, reward, terminated, truncated, info = env.step(action_idx)

    assert env.time_remaining == initial_time - 1

    assert env.shares_remaining < initial_shares
    assert info["shares_sold"] > 0

    assert isinstance(reward, float)

    assert next_state.shape == (3,)


def test_liquidation_constraint(env):
    """Test that the environment forces liquidation at the last step."""
    env.reset()

    env.time_remaining = 1
    env.shares_remaining = 500

    action_idx = 0
    _, _, terminated, _, info = env.step(action_idx)

    assert info["shares_sold"] == 500
    assert env.shares_remaining == 0
    assert terminated is True


def test_agent_network_shape(agent):
    """Test that the DQN outputs the correct number of Q-values."""
    state = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

    q_values = agent.policy_net(state_t)

    assert q_values.shape == (1, agent.action_dim)


def test_replay_buffer(agent):
    """Test that the replay buffer stores and samples correctly."""
    from rl_order_execution.agent import ReplayBuffer

    buffer = ReplayBuffer(capacity=10, device=agent.device)

    state = np.zeros(3)
    next_state = np.zeros(3)
    buffer.push(state, 0, 1.0, next_state, False)

    assert len(buffer) == 1

    batch = buffer.sample(1)
    states, actions, rewards, next_states, dones = batch

    assert states.shape == (1, 3)
    assert rewards.item() == 1.0
