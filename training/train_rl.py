from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_env.elevatorGym import ElevatorEnv
import time
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import torch
import os
import re
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from torch.utils.tensorboard import SummaryWriter
import numpy as np

CHECKPOINT_DIR = "../model/checkpoints/"
MODEL_DIR = "../model/"
DATA_DIR = "../data/"
MODEL_PREFIX = "ppo_multi_elevator"
TOTAL_TIMESTEPS = 500_000
TENSORBOARD_LOG_DIR = "../logs/tensorboard_logs/"
TENSORBOARD_RUN_NAME = "run1"

def loadFloorDistribution(fileName, numFloors, simTime):
    path = os.path.join(DATA_DIR, fileName)
    try:
        with open(path, "r") as file:
            matrix = [list(map(float, line.split()[:numFloors])) for line in file if line.strip()]
    except FileNotFoundError:
        print(f"{path} not found. Using uniform distribution.") 
        matrix = [] 
 
    expected_columns = numFloors 
    expected_rows = simTime 
 
    valid = ( 
        len(matrix) >= expected_rows 
        and all(len(row) == expected_columns for row in matrix) 
        and all(abs(sum(row) - 1.0) < 1e-9 for row in matrix) 
    ) 
 
    if not valid: 
        uniform_row = [1.0 / expected_columns] * expected_columns
        matrix = [uniform_row[:] for _ in range(expected_rows)]
        print("Invalid distribution file. Using uniform distribution.")
    print(matrix)
    return matrix

def loadAverageArrivalTimes(fileName, default_time, simTime):
    path = os.path.join(DATA_DIR, fileName)
    try:
        with open(path, "r") as file:
            line = file.readline()
            matrix = list(map(float, line.split()))
    except FileNotFoundError:
        print(f"{path} not found. Using default average arrival times.")
        matrix = [default_time] * simTime

    if len(matrix) < simTime:
        print(f"Invalid data length. Expected {simTime}, got {len(matrix)}.")
        matrix = [default_time] * simTime

    return matrix

def make_env(seed=None):
    floorDistribution = loadFloorDistribution("../data/floorDistribution.txt", 10, 300)
    averageArrivalTimes = loadAverageArrivalTimes("../data/averageArivalTimes.txt", 20.0, 300)
    env = ElevatorEnv(
        floorDistribution,
        averageArrivalTimes,
        num_floors=10,
        num_elevators=1,
        elevator_capacity=5,
        simulation_time=300,
        fahrzeit=1.0,
        halt_zeit=0.5,
        elevator_check_interval=1.0, 
        scanning=False
    )
    if seed is not None:
        env.seed(seed)
    env = ActionMasker(env, action_mask_fn=lambda env: env.getMask())
    return Monitor(env)

def get_latest_checkpoint(path, prefix):
    if not os.path.exists(path):
        return None

    files = os.listdir(path)
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_steps\.zip")
    checkpoints = [(int(match.group(1)), os.path.join(path, file))
                   for file in files if (match := pattern.match(file))]

    return max(checkpoints, key=lambda x: x[0])[1] if checkpoints else None

def load_model_or_initialize(env):
    checkpoint_path = get_latest_checkpoint(CHECKPOINT_DIR, MODEL_PREFIX)
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        return MaskablePPO.load(checkpoint_path, env=env, device="cpu", verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR), True
    else:
        print("No checkpoint found. Initializing new model.")
        return MaskablePPO("MlpPolicy", env, device="cpu",  verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR), False

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"Training started: {self.total_timesteps} total timesteps")

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            print(f"Step {self.num_timesteps}/{self.total_timesteps} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        return True

class ExtendedLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        norms = []
        for p in self.model.policy.parameters():
            if p.grad is not None:
                norms.append(torch.norm(p.grad.detach()))
        if norms:
            grad_norm = torch.norm(torch.stack(norms)).item()
            self.writer.add_scalar("gradients/global_norm", grad_norm, self.num_timesteps)

        actions = self.locals.get("actions")
        if actions is not None:
            self.writer.add_histogram("actions/distribution", np.array(actions), self.num_timesteps)
        
        if hasattr(self.training_env.envs[0].unwrapped, "get_metrics"):
            metrics = self.training_env.envs[0].unwrapped.get_metrics()
            for key, value in metrics.items():
                self.writer.add_scalar(f"env/{key}", value, self.num_timesteps)

        return True

    def _on_training_end(self):
        self.writer.close()

def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = make_env(seed=0)
    model, was_loaded = load_model_or_initialize(env)

    progress_callback = ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_PREFIX
    )
    extended_logging = ExtendedLoggingCallback(os.path.join(TENSORBOARD_LOG_DIR, TENSORBOARD_RUN_NAME))
    callback = CallbackList([progress_callback, checkpoint_callback, extended_logging])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=not was_loaded,
        tb_log_name=TENSORBOARD_RUN_NAME
    )

    model.save(os.path.join(MODEL_DIR, MODEL_PREFIX))
    print(f"Model saved as '{os.path.join(MODEL_DIR, MODEL_PREFIX)}.zip'")

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    train_and_save_model()
