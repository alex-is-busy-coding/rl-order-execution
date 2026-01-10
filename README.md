# Deep Reinforcement Learning for Order Execution

This project implements a Deep Q-Network (DQN) agent designed to execute large financial trade orders optimally. The agent learns to balance the trade-off between **market impact** (slippage caused by trading too fast) and **market risk** (price volatility risk from holding inventory too long).

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Highly Recommended for dependency resolution)
- [Docker](https://www.docker.com/) (Optional, for isolated execution)

### Setup

- **Clone the repository:**

    ```bash
    git clone https://github.com/alex-is-busy-coding/rl-order-execution.git
    cd rl-order-execution
    ```

- **Install dependencies and run:**
    ```bash
    make run
    ```

This executes the training loop, compares the agent against a TWAP benchmark, and saves a trajectory plot to `execution_analysis.png`.

### Training Tracking

The training loop automatically logs loss, reward, and epsilon decay to TensorBoard.

```bash
make tensorboard
```

Open http://localhost:6006 to view the metrics.

### Docker Support

To run the simulation in a completely isolated environment:

- **Build the image:**

    ```bash
    make docker-build
    ```

- **Run the container:** Mounts the local directory to capture output artifacts.

    ```bash
    make docker-run
    ```

## Project Structure

```
rl-order-execution/
├── .github/workflows/
│   ├── ci.yml           # CI pipeline (Lint, Test, Type-Check)
│   └── update-docs.yml  # Auto-update README config table
├── config/
│   └── config.yaml      # Runtime configuration parameters
├── src/
│   └── rl_order_execution/
│       ├── agent.py         # DQN Agent & ReplayBuffer implementation
│       ├── settings.py      # Pydantic configuration & validation
│       ├── environment.py   # Custom Gymnasium Market Environment
│       ├── evaluation.py    # TWAP comparison & plotting logic
│       └── training.py      # Core training loop with TensorBoard
├── tests/               # Pytest suite
├── Dockerfile           # Container definition
├── Makefile             # Automation commands
├── pyproject.toml       # Dependencies (uv)
├── README.md            # Documentation
└── main.py              # Application entry point
```

## Development Workflow

We use `make` to standardize development tasks and ensure code quality. 

| Command           | Description |
| ----------------- | ----------- |
| `make run`          | Run the simulation
| `make tensorboard`  | Launch TensorBoard server
| `make check`        | *Recommended.* Run all quality checks (lint + type-check + test)
| `make test`        | Run unit tests
| `make lint`         | Check code style
| `make type-check`   | Run static type checking with mypy
| `make format`       | Auto-format code
| `make docs`         | Locally update README config table
| `make install`      | Install base dependencies
| `make install-dev`  | Install all dev dependencies
| `make docker-build` | Build the Docker image
| `make docker-run`   | Run the Docker container
| `make clean`        | Remove virtualenv, caches, and plots

Run `make help` in your terminal to see the full list of available commands.

## Configuration

Configuration is managed via `pydantic-settings`. You can override defaults using environment variables or by editing `config/config.yaml`.

<!-- settings-start -->
### `logging` settings


| Name | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `RL_LOGGING__LOG_LEVEL` | No | `INFO` | Logging verbosity level.<br><br>**Possible values:**<br>`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### `rl` settings


| Name | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `RL_RL__BATCH_SIZE` | No | `64` | Training batch size. |
| `RL_RL__EPISODES` | No | `500` | Total training episodes. |
| `RL_RL__EPSILON_DECAY` | No | `0.995` | Epsilon decay factor. |
| `RL_RL__EPSILON_END` | No | `0.01` | Minimum exploration rate. |
| `RL_RL__EPSILON_START` | No | `1.0` | Initial exploration rate. |
| `RL_RL__GAMMA` | No | `0.99` | Discount factor. |
| `RL_RL__LR` | No | `0.001` | Learning rate. |
| `RL_RL__MEMORY_SIZE` | No | `10000` | Replay buffer size. |
| `RL_RL__TARGET_UPDATE` | No | `10` | Episodes between target updates. |

### `simulation` settings


| Name | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `RL_SIMULATION__ACTION_MULTIPLIERS` | No | `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]` | Discrete multipliers of the average execution rate. |
| `RL_SIMULATION__DRIFT` | No | `0.0` | Price drift (mu). |
| `RL_SIMULATION__LIQUIDITY_PARAM` | No | `0.01` | Permanent market impact (alpha). |
| `RL_SIMULATION__SEED` | No | `42` | Random seed for reproducibility. |
| `RL_SIMULATION__START_PRICE` | No | `100.0` | Initial market price. |
| `RL_SIMULATION__TEMP_IMPACT_PARAM` | No | `0.05` | Temporary market impact (beta). |
| `RL_SIMULATION__TIME_HORIZON` | No | `50` | Total duration (time steps). |
| `RL_SIMULATION__TOTAL_SHARES` | No | `1000` | Total number of shares to liquidate. |
| `RL_SIMULATION__VOLATILITY` | No | `0.002` | Price volatility (sigma). |
<!-- settings-end -->

## Limitations & Future Improvements

While this project demonstrates a robust RL pipeline, it makes certain simplifying assumptions common in initial research but limiting for production deployment.

### 1. Discrete vs. Continuous Control (DQN vs. PPO/SAC)

**Limitation:** The current agent uses a Deep Q-Network (DQN), which necessitates a discrete action space. Execution rates are quantized into specific bins (e.g., 0.5x, 1.0x, 2.0x TWAP).

**Future Improvement:** Implement Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC). These algorithms support continuous action spaces, allowing the agent to output precise execution rates without artificial quantization buckets.

## License

This project is licensed under the MIT License. 

See the [LICENSE](LICENSE) file for details.