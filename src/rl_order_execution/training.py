import time
import numpy as np
import logging
from typing import List
from torch.utils.tensorboard import SummaryWriter

from rl_order_execution.settings import get_settings
from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent

logger = logging.getLogger(__name__)


def train_agent(env: OrderExecutionEnv, agent: DQNAgent) -> List[float]:
    """
    Executes the training loop for the Deep Q-Network agent.
    """
    settings = get_settings()
    rl_conf = settings.rl

    writer = SummaryWriter(
        log_dir=f"runs/dqn_episodes_{rl_conf.episodes}_lr_{rl_conf.lr}"
    )

    rewards_history = []
    logger.info(f"Starting Training for {rl_conf.episodes} episodes...")
    start_time = time.time()

    global_step = 0

    for episode in range(rl_conf.episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_losses = []

        while not (terminated or truncated):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            global_step += 1

        agent.update_epsilon()

        if episode % rl_conf.target_update == 0:
            agent.update_target_network()

        rewards_history.append(episode_reward)

        writer.add_scalar("Train/Reward", episode_reward, episode)
        writer.add_scalar("Train/Epsilon", agent.epsilon, episode)
        if episode_losses:
            writer.add_scalar("Train/Avg_Loss", np.mean(episode_losses), episode)

        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            logger.info(
                f"Episode {episode + 1:03d} | Avg Reward: {avg_rew:.2f} | Epsilon: {agent.epsilon:.2f}"
            )

    logger.info(f"Training finished in {time.time() - start_time:.1f}s")
    writer.close()
    return rewards_history
