import torch
import numpy as np
import random
import yaml
from typing import Tuple, Type, Dict, Any, List, Annotated
from functools import lru_cache
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A custom settings source that loads configuration from a 'config.yaml' file
    if it exists in the current working directory.
    """

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        try:
            with open("config/config.yaml", "r") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}


class YamlFileSource(PydanticBaseSettingsSource):
    """
    A custom settings source that loads configuration from a specific YAML file.
    """

    def __init__(self, settings_cls: Type[BaseSettings], filepath: str):
        super().__init__(settings_cls)
        self.filepath = filepath

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}


class SimulationSettings(BaseModel):
    """Settings related to market simulation and execution constraints."""

    seed: Annotated[int, Field(description="Random seed for reproducibility.")] = 42
    total_shares: Annotated[
        int, Field(description="Total number of shares to liquidate.")
    ] = 1000
    time_horizon: Annotated[int, Field(description="Total duration (time steps).")] = 50
    start_price: Annotated[float, Field(description="Initial market price.")] = 100.0
    volatility: Annotated[float, Field(description="Price volatility (sigma).")] = 0.002
    drift: Annotated[float, Field(description="Price drift (mu).")] = 0.0
    liquidity_param: Annotated[
        float, Field(description="Permanent market impact (alpha).")
    ] = 0.01
    temp_impact_param: Annotated[
        float, Field(description="Temporary market impact (beta).")
    ] = 0.05
    action_multipliers: Annotated[
        List[float],
        Field(description="Discrete multipliers of the average execution rate."),
    ] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    @field_validator("total_shares", "time_horizon", "start_price")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("volatility")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Volatility cannot be negative")
        return v


class RLSettings(BaseModel):
    """Hyperparameters for the Reinforcement Learning agent."""

    gamma: Annotated[float, Field(description="Discount factor.")] = 0.99
    batch_size: Annotated[int, Field(description="Training batch size.")] = 64
    lr: Annotated[float, Field(description="Learning rate.")] = 0.001
    epsilon_start: Annotated[float, Field(description="Initial exploration rate.")] = (
        1.0
    )
    epsilon_end: Annotated[float, Field(description="Minimum exploration rate.")] = 0.01
    epsilon_decay: Annotated[float, Field(description="Epsilon decay factor.")] = 0.995
    target_update: Annotated[
        int, Field(description="Episodes between target updates.")
    ] = 10
    memory_size: Annotated[int, Field(description="Replay buffer size.")] = 10000
    episodes: Annotated[int, Field(description="Total training episodes.")] = 500

    @field_validator("gamma", "epsilon_start", "epsilon_end", "epsilon_decay")
    @classmethod
    def must_be_between_zero_and_one(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("Must be between 0 and 1")
        return v


class OptimizationSettings(BaseModel):
    """Settings defining the search space for hyperparameter optimization."""

    study_name: Annotated[
        str,
        Field(
            description="Name of the Optuna study. Change this to start a new experiment."
        ),
    ] = "rl_order_execution_v1"

    lr_min: Annotated[float, Field(description="Minimum learning rate to test.")] = 1e-5
    lr_max: Annotated[float, Field(description="Maximum learning rate to test.")] = 1e-2

    batch_sizes: Annotated[
        List[int], Field(description="List of batch sizes to test.")
    ] = [32, 64, 128]

    gamma_min: Annotated[float, Field(description="Minimum discount factor.")] = 0.90
    gamma_max: Annotated[float, Field(description="Maximum discount factor.")] = 0.9999

    n_trials: Annotated[int, Field(description="Number of Optuna trials to run.")] = 20
    tuning_episodes: Annotated[
        int, Field(description="Episodes per trial (shorter than production run).")
    ] = 500

    @field_validator("lr_min", "lr_max", "n_trials", "tuning_episodes")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("gamma_min", "gamma_max")
    @classmethod
    def must_be_between_zero_and_one(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("Must be between 0 and 1")
        return v

    @field_validator("batch_sizes")
    @classmethod
    def must_contain_positive_integers(cls, v: List[int]) -> List[int]:
        if not all(i > 0 for i in v):
            raise ValueError("Batch sizes must be positive integers")
        return v


class LoggingSettings(BaseModel):
    """Settings for application logging."""

    log_level: Annotated[
        str,
        Field(
            description="Logging verbosity level.",
            json_schema_extra={
                "possible_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            },
        ),
    ] = "INFO"

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in levels:
            raise ValueError(f"Log level must be one of {levels}")
        return v.upper()


class Settings(BaseSettings):
    """
    Root Configuration.
    Access fields via settings.simulation, settings.rl, or settings.logging.
    """

    model_config = SettingsConfigDict(
        env_prefix="RL_", env_nested_delimiter="__", env_file_encoding="utf-8"
    )

    simulation: SimulationSettings = Field(default_factory=lambda: SimulationSettings())
    rl: RLSettings = Field(default_factory=lambda: RLSettings())
    optimization: OptimizationSettings = Field(
        default_factory=lambda: OptimizationSettings()
    )
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings())

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlFileSource(settings_cls, "config/best_params.yaml"),
            YamlFileSource(settings_cls, "config/config.yaml"),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
