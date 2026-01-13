import optuna
import logging
import torch
import yaml
import os
from gymnasium import spaces

from rl_order_execution.settings import get_settings, set_seeds
from rl_order_execution.environment import OrderExecutionEnv
from rl_order_execution.agent import DQNAgent
from rl_order_execution.training import train_agent
from rl_order_execution.evaluation import evaluate_agent

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    """
    settings = get_settings()
    opt_conf = settings.optimization

    lr = trial.suggest_float("lr", opt_conf.lr_min, opt_conf.lr_max, log=True)
    batch_size = trial.suggest_categorical("batch_size", opt_conf.batch_sizes)
    gamma = trial.suggest_float("gamma", opt_conf.gamma_min, opt_conf.gamma_max)
    episodes = opt_conf.tuning_episodes

    settings.rl.lr = lr
    settings.rl.batch_size = batch_size
    settings.rl.gamma = gamma
    settings.rl.episodes = episodes

    set_seeds(settings.simulation.seed)
    device = torch.device("cpu")

    try:
        env = OrderExecutionEnv(settings)
        if not isinstance(env.action_space, spaces.Discrete):
            return -float("inf")

        agent = DQNAgent(
            state_dim=3, action_space=env.action_space, settings=settings, device=device
        )

        train_agent(env, agent, optuna_trial=trial)

        twap_avg, rl_avg = evaluate_agent(env, agent, num_trials=10)

        if twap_avg == 0:
            return 0.0
        improvement = (rl_avg - twap_avg) / twap_avg * 100
        return float(improvement)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial failed with error: {e}")
        return -999.0


def save_best_params(study: optuna.Study):
    """Saves the best parameters to config/best_params.yaml"""
    logger.info("Saving best parameters to config/best_params.yaml...")

    best_config = {
        "rl": {
            "lr": study.best_params["lr"],
            "batch_size": study.best_params["batch_size"],
            "gamma": study.best_params["gamma"],
            "episodes": 2000,
        }
    }

    os.makedirs("config", exist_ok=True)
    with open("config/best_params.yaml", "w") as f:
        yaml.dump(best_config, f)
    logger.info("Best parameters saved successfully.")


def run_optimization():
    logger.info("Starting Hyperparameter Optimization with Optuna...")

    settings = get_settings()
    n_trials = settings.optimization.n_trials

    os.makedirs("db", exist_ok=True)

    storage_name = "sqlite:///db/optuna_study.db"
    study_name = settings.optimization.study_name

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(f"Running {n_trials} trials (Storage: {storage_name})...")
    study.optimize(objective, n_trials=n_trials)

    logger.info("-" * 50)
    logger.info("Optimization Finished.")
    logger.info(f"Best Value (Improvement %): {study.best_value:.4f}%")
    logger.info("Best Params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 50)

    save_best_params(study)


if __name__ == "__main__":
    run_optimization()
