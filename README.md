
# 🚀 Multi-Elevator Reinforcement Learning Simulation

This project simulates a building with multiple elevators and trains a reinforcement learning (RL) agent to control elevator movements efficiently using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [SimPy](https://simpy.readthedocs.io/).

---

## 📁 Project Structure

### 🧠 Training

- **`train_rl.py`**  
  Entry point for training the RL agent. Loads the environment, sets up callbacks and logging, and saves the final model under `model/`.

- **`averageArivalTimes.txt`**  
  Time-based expected arrival rates of people per simulation step (used for people spawning).

- **`floorDistribution.txt`** *(optional)*  
  Matrix of destination probabilities for each second and floor. If missing or invalid, a uniform distribution is used.

---

### 🏢 Gym Environment

- **`gym_env/elevatorGym.py`**  
  Custom `ElevatorEnv` built with [gymnasium](https://gymnasium.farama.org/).  
  Handles:
  - Discrete action space (per-elevator actions: up/down/stop)
  - Complex observation space (elevator states, target floors, queues)
  - Action masking for legal moves only
  - Reward shaping based on travel and waiting times

---

### 🏗️ Simulation Core (`env/`)

- **`House.py`**  
  Coordinates floors and elevators. Runs the core SimPy environment.

- **`Aufzug.py`**  
  Models individual elevators and their movement/state logic.

- **`Floor.py`**  
  Represents a single floor including the up/down queues.

- **`Person.py`**  
  Simulates a person with a target floor, arrival time, and waiting behavior.

- **`Dispatcher.py`**  
  SCAN-style dispatcher that polls floors for elevator requests.

- **`ElevatorController.py`**  
  Handles logic for responding to elevator requests and queues.

---

### 🖥️ Simulation GUI

- **`SimulationGUI.py`**  
  Visual Tkinter interface to run and interact with the elevator simulation manually.

- **`Simulation_plot.py`**  
  Contains plotting logic for elevator and passenger states using Matplotlib.

- **`main.py`**  
  Entry point to launch the GUI. Sets up logging and the Tkinter window.

---

## 📦 Python Modules

This project uses a modular structure with two internal packages:

- `gym_env`: for the custom Gym environment.
- `env`: for the simulation logic.

Make sure each subfolder contains an `__init__.py` file to enable proper imports.

---

## 📂 Other Folders

- **`logs/`**  
  Stores environment and GUI logs.

- **`checkpoints/`**  
  Auto-saved model checkpoints during training.

- **`tensorboard_logs/`**  
  Stores logs for TensorBoard visualizations.

- **`model/`**  
  The final saved model after training.

---

## ▶️ How to Run

### Train the model:

```bash
python train_rl.py
