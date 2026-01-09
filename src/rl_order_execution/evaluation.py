import numpy as np
import matplotlib.pyplot as plt
import logging

from rl_order_execution.settings import get_settings
from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent

logger = logging.getLogger(__name__)


def evaluate_agent(env: OrderExecutionEnv, agent: DQNAgent, num_trials: int = 100):
    """
    Compares the trained RL agent against a standard TWAP benchmark.
    """
    logger.info(f"\nEvaluating: RL Agent vs TWAP ({num_trials} trials)...")

    twap_idx = min(
        range(len(env.action_multipliers)),
        key=lambda i: abs(env.action_multipliers[i] - 1.0),
    )

    rl_prices = []
    twap_prices = []

    for _ in range(num_trials):
        env.reset()
        terminated = False
        total_rev, total_sold = 0.0, 0
        while not terminated:
            _, _, terminated, _, info = env.step(twap_idx)
            total_rev += info["revenue"]
            total_sold += info["shares_sold"]

        twap_avg = total_rev / total_sold if total_sold > 0 else 0.0
        twap_prices.append(twap_avg)

        state, _ = env.reset()
        terminated = False
        total_rev, total_sold = 0.0, 0
        while not terminated:
            action = agent.select_action(state, is_eval=True)
            next_state, _, terminated, _, info = env.step(action)
            total_rev += info["revenue"]
            total_sold += info["shares_sold"]
            state = next_state

        rl_avg = total_rev / total_sold if total_sold > 0 else 0.0
        rl_prices.append(rl_avg)

    mean_twap = np.mean(twap_prices)
    mean_rl = np.mean(rl_prices)
    improvement = (mean_rl - mean_twap) / mean_twap * 100

    logger.info(f"TWAP Avg Exec Price: ${mean_twap:.2f}")
    logger.info(f"RL Agent Avg Exec Price: ${mean_rl:.2f}")
    logger.info(f"Agent Improvement: {improvement:+.4f}%")

    return mean_twap, mean_rl


def visualize_trajectory(
    env: OrderExecutionEnv, agent: DQNAgent, filename: str = "execution_analysis.png"
):
    """
    Runs a single demonstration episode and plots the inventory/price trajectory.
    """
    settings = get_settings()

    state, _ = env.reset()
    rl_inventory = [env.shares_remaining]
    rl_market_price = [env.current_price]
    terminated = False

    while not terminated:
        action = agent.select_action(state, is_eval=True)
        next_state, _, terminated, _, _ = env.step(action)
        state = next_state
        rl_inventory.append(env.shares_remaining)
        rl_market_price.append(env.current_price)

    time_steps = len(rl_inventory)
    twap_inventory = [
        max(
            0,
            settings.simulation.total_shares
            - (
                i
                * (settings.simulation.total_shares / settings.simulation.time_horizon)
            ),
        )
        for i in range(time_steps)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(twap_inventory, label="TWAP (Benchmark)", linestyle="--", color="gray")
    ax1.plot(rl_inventory, label="RL Agent (DQN)", color="blue", linewidth=2)
    ax1.set_title("Inventory Trajectory: RL vs TWAP")
    ax1.set_ylabel("Shares Remaining")
    ax1.set_xlabel("Time Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(rl_market_price, label="Market Price (RL Episode)", color="green")
    ax2.set_title("Underlying Market Price Movement")
    ax2.set_ylabel("Price ($)")
    ax2.set_xlabel("Time Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Plot saved to '{filename}'")
