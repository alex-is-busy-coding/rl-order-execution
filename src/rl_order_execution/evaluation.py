import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import seaborn as sns
import os

from rl_order_execution.settings import get_settings
from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent

logger = logging.getLogger(__name__)


def calculate_is(env: OrderExecutionEnv, revenue: float) -> float:
    """
    Calculates Implementation Shortfall (IS).
    IS = Paper Value (Arrival Price * Shares) - Realized Revenue
    """
    paper_value = env.total_shares * env.start_price
    return paper_value - revenue


def evaluate_agent(env: OrderExecutionEnv, agent: DQNAgent, num_trials: int = 100):
    """
    Compares RL Agent vs TWAP using paired trials (same seed).
    Metrics: IS, Win Rate, IR, Basis Points (bps), VaR 95%.
    """
    logger.info(f"\nEvaluating: RL Agent vs TWAP ({num_trials} paired trials)...")

    twap_idx = min(
        range(len(env.action_multipliers)),
        key=lambda i: abs(env.action_multipliers[i] - 1.0),
    )

    results = []

    paper_value = env.total_shares * env.start_price

    for i in range(num_trials):
        seed = 42000 + i

        env.reset(seed=seed)
        terminated = False
        twap_rev = 0.0
        while not terminated:
            _, _, terminated, _, info = env.step(twap_idx)
            twap_rev += info["revenue"]
        twap_is = calculate_is(env, twap_rev)

        state, _ = env.reset(seed=seed)
        terminated = False
        rl_rev = 0.0
        while not terminated:
            action = agent.select_action(state, is_eval=True)
            next_state, _, terminated, _, info = env.step(action)
            rl_rev += info["revenue"]
            state = next_state
        rl_is = calculate_is(env, rl_rev)

        is_savings = twap_is - rl_is

        results.append(
            {
                "episode": i,
                "twap_is": twap_is,
                "rl_is": rl_is,
                "is_savings": is_savings,
                "rl_beat_twap": rl_is < twap_is,
            }
        )

    df = pd.DataFrame(results)

    avg_twap_is = df["twap_is"].mean()
    avg_rl_is = df["rl_is"].mean()
    avg_savings = df["is_savings"].mean()
    std_savings = df["is_savings"].std()

    avg_savings_bps = (avg_savings / paper_value) * 10000

    ir = avg_savings / std_savings if std_savings != 0 else 0.0

    win_rate = df["rl_beat_twap"].mean() * 100

    var_95 = np.percentile(df["is_savings"], 5)

    logger.info("-" * 50)
    logger.info("QUANTITATIVE METRICS (Implementation Shortfall)")
    logger.info("-" * 50)
    logger.info(f"Avg TWAP IS:       ${avg_twap_is:8.2f}")
    logger.info(f"Avg RL IS:         ${avg_rl_is:8.2f}")
    logger.info(f"Avg Savings:       ${avg_savings:8.2f} (Active Return)")
    logger.info(f"Avg Savings (bps): {avg_savings_bps:8.2f} bps")
    logger.info(f"Information Ratio: {ir:8.4f}")
    logger.info(f"Win Rate:          {win_rate:8.1f}%")
    logger.info(f"VaR 95% (Savings): ${var_95:8.2f} (Tail Risk)")
    logger.info("-" * 50)

    os.makedirs("output", exist_ok=True)

    plot_distribution(df, "output/is_distribution.png")

    return avg_twap_is, avg_rl_is


def plot_distribution(df: pd.DataFrame, filename: str):
    """Plots the distribution of Implementation Shortfall Savings."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["is_savings"], kde=True, bins=30, color="blue", alpha=0.6)

    mean_savings = df["is_savings"].mean()
    var_95 = np.percentile(df["is_savings"], 5)

    plt.axvline(0, color="red", linestyle="--", label="TWAP Baseline")
    plt.axvline(
        mean_savings, color="green", linestyle="-", label=f"Mean (${mean_savings:.2f})"
    )
    plt.axvline(var_95, color="orange", linestyle=":", label=f"VaR 95% (${var_95:.2f})")

    plt.title("Distribution of RL Agent Savings vs TWAP (Active Return)")
    plt.xlabel(
        "Cost Savings ($) - Positive = Outperformance, Negative = Underperformance"
    )
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Distribution plot saved to '{filename}'")


def visualize_trajectory(
    env: OrderExecutionEnv,
    agent: DQNAgent,
    filename: str = "output/execution_analysis.png",
):
    """
    Runs a single demonstration episode and plots the inventory/price trajectory.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    settings = get_settings()
    sim_conf = settings.simulation

    state, _ = env.reset(seed=999)
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
            sim_conf.total_shares
            - (i * (sim_conf.total_shares / sim_conf.time_horizon)),
        )
        for i in range(time_steps)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rl_inventory, label="RL Agent (DQN)", color="blue", linewidth=4, alpha=0.6)
    ax1.plot(
        twap_inventory,
        label="TWAP (Benchmark)",
        linestyle="--",
        color="orange",
        linewidth=2,
    )

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
    logger.info(f"Trajectory plot saved to '{filename}'")
