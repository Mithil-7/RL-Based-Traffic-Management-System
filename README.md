# SIH 2025 Traffic Management

This repository contains a Python prototype for adaptive traffic signal control using a Deep Q-Network (DQN) approach.

## Repository Contents

- `import random.py`: Main script that defines:
  - `DynamicTrafficEnvironment` for simulating road traffic states
  - `DQNAgent` for reinforcement learning-based traffic signal decisions
  - Training loop for running simulation episodes

## Prerequisites

- Python 3.8+
- `numpy`
- `tensorflow`
- `matplotlib`

Install dependencies:

```bash
pip install numpy tensorflow matplotlib
```

## Run the Project

```bash
python "import random.py"
```

## Notes

- The current implementation is a simulation/prototype.
- Logging is printed directly to the console during training and road-state updates.
