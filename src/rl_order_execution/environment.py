import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Any, Optional
import logging

from rl_order_execution.settings import Settings

logger = logging.getLogger(__name__)


class OrderExecutionEnv(gym.Env):
    """
    A Custom Gymnasium Environment for Optimal Trade Execution.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, settings: Settings):
        super(OrderExecutionEnv, self).__init__()
        self.settings = settings
        self.total_shares = settings.simulation.total_shares
        self.time_horizon = settings.simulation.time_horizon
        self.start_price = settings.simulation.start_price

        self.avg_rate = self.total_shares / self.time_horizon
        self.action_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

        self.action_space = spaces.Discrete(len(self.action_multipliers))

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf], dtype=np.float32),
            high=np.array([1, 1, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        logger.info(
            f"Initialized OrderExecutionEnv: Shares={self.total_shares}, Horizon={self.time_horizon}"
        )

        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.shares_remaining = self.total_shares
        self.time_remaining = self.time_horizon
        self.current_price = self.start_price
        self.prev_price = self.start_price

        self.price_history = [self.start_price]
        self.execution_history: List[int] = []

        logger.debug("Environment reset")
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        normalized_shares = self.shares_remaining / self.total_shares
        normalized_time = self.time_remaining / self.time_horizon

        if self.prev_price > 0:
            price_trend = (self.current_price - self.prev_price) / self.prev_price
        else:
            price_trend = 0.0

        return np.array(
            [normalized_shares, normalized_time, price_trend], dtype=np.float32
        )

    def _update_market_price(self) -> float:
        shock = np.random.normal(0, 1)
        change = self.current_price * (
            self.settings.simulation.drift + self.settings.simulation.volatility * shock
        )
        return self.current_price + change

    def _calculate_reward(
        self, shares_sold: int, temp_impact: float, inventory_risk: float
    ) -> float:
        slippage_cost = shares_sold * temp_impact
        return -slippage_cost - inventory_risk

    def step(
        self, action_idx: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        multiplier = self.action_multipliers[action_idx]
        shares_to_sell = int(self.avg_rate * multiplier)

        shares_to_sell = min(shares_to_sell, self.shares_remaining)

        if self.time_remaining == 1:
            shares_to_sell = self.shares_remaining

        mid_price = self._update_market_price()

        temp_impact = self.settings.simulation.temp_impact_param * shares_to_sell
        exec_price = mid_price - temp_impact
        perm_impact = self.settings.simulation.liquidity_param * shares_to_sell
        next_price = mid_price - perm_impact

        inventory_risk = 0.01 * (self.shares_remaining**2) / self.total_shares
        reward = self._calculate_reward(shares_to_sell, temp_impact, inventory_risk)

        revenue = shares_to_sell * exec_price
        self.shares_remaining -= shares_to_sell
        self.time_remaining -= 1
        self.prev_price = self.current_price
        self.current_price = next_price

        self.price_history.append(self.current_price)
        self.execution_history.append(shares_to_sell)

        terminated = (self.time_remaining == 0) or (self.shares_remaining == 0)
        truncated = False

        info = {
            "revenue": revenue,
            "avg_exec_price": exec_price if shares_to_sell > 0 else 0.0,
            "shares_sold": shares_to_sell,
        }

        return self._get_state(), reward, terminated, truncated, info
