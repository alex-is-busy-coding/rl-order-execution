import logging
import torch

from rl_order_execution.settings import get_settings, set_seeds
from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent
from rl_order_execution.training import train_agent
from rl_order_execution.evaluation import evaluate_agent, visualize_trajectory

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.logging.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    set_seeds(settings.simulation.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing RL Order Execution System...")
    env = OrderExecutionEnv(settings)

    agent = DQNAgent(
        state_dim=3, action_space=env.action_space, settings=settings, device=device
    )

    train_agent(env, agent)
    evaluate_agent(env, agent)
    visualize_trajectory(env, agent)


if __name__ == "__main__":
    main()
