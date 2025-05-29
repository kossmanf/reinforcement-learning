from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from elevatorGym import ElevatorEnv
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import torch
import os
import re
from stable_baselines3.common.vec_env import SubprocVecEnv

CHECKPOINT_DIR = "./checkpoints/"
MODEL_PREFIX = "ppo_multi_elevator"

def make_env():
    floorDistribution = loadFloorDistribution("floorDistribution.txt", 10, 300)
    averrageArivalTimes = loadAverageArivalTimes("averageArivalTimes.txt", 20.0, 300)
    return ElevatorEnv(
        floorDistribution,
        averrageArivalTimes,
        num_floors=10,
        num_elevators=3,
        elevator_capacity=4,
        simulation_time=300,
        fahrzeit=1.0,
        halt_zeit=0.5,
        elevator_check_interval=1.0
    )

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

def make_env_fn(rank):
    def _init():
        return make_env()
    return _init

def train_and_save_model():
    env = DummyVecEnv([make_env])
    model = model = load_model_or_initialize(env)

    num_cpu = 4
    env = SubprocVecEnv([make_env_fn(i) for i in range(num_cpu)])

    # Create progress bar and checkpoint callbacks
    total_timesteps = 100_000
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_multi_elevator"
    )

    callback = CallbackList([progress_callback, checkpoint_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_multi_elevator")
    print("Model saved as 'ppo_multi_elevator.zip'")

def get_latest_checkpoint(path, prefix):
    """Find the latest checkpoint file based on step number."""
    if not os.path.exists(path):
        return None

    files = os.listdir(path)
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_steps\.zip")

    # Find files that match the pattern and extract the step number
    checkpoints = [(int(match.group(1)), os.path.join(path, file))
                   for file in files if (match := pattern.match(file))]

    if not checkpoints:
        return None

    # Return path to checkpoint with highest step count
    _, latest_path = max(checkpoints, key=lambda x: x[0])
    return latest_path
    
def load_model_or_initialize(env):
    checkpoint_path = get_latest_checkpoint(CHECKPOINT_DIR, MODEL_PREFIX)
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        return PPO.load(checkpoint_path, env=env, device="cuda")
    else:
        print("No checkpoint found. Initializing new model.")
        return PPO("MlpPolicy", env, verbose=1, device="cuda")

def loadAverageArivalTimes(fileName, AVERAGE_ARRIVAL_TIME, simTime):
    """
    Loads the average person arrival times for each timestep.
    If file missing or invalid -> fallbacks to default 10s arrival times.
    """
    try:
        with open(fileName, "r") as file:
            line = file.readline()
            matrix = list(map(float, line.split()))
    except FileNotFoundError:
        print("File not found. Using default average arrival times of 10.")
        matrix = [AVERAGE_ARRIVAL_TIME] * simTime

    expected_length = simTime
    if len(matrix) < expected_length:
        print(f"Invalid data length. Expected {expected_length}, got {len(matrix)}.")
        matrix = [AVERAGE_ARRIVAL_TIME] * expected_length

    return matrix

def loadFloorDistribution(fileName, numFloors, simTime):
    """
    Loads the floor target probability distribution for each timestep.
    If file missing or invalid -> fallbacks to uniform distribution.
    """
    try:
        with open(fileName, "r") as file:
            matrix = [list(map(float, line.split())) for line in file if line.strip()]
    except FileNotFoundError:
        print("File not found. Using uniform distribution.")
        matrix = []

    expected_columns = numFloors
    expected_rows = simTime

    valid = True
    if len(matrix) < expected_rows:
        valid = False
    elif not all(len(row) >= expected_columns for row in matrix):
        valid = False
    elif not all(abs(sum(row) - 1.0) < 1e-9 for row in matrix):
        valid = False

    if not valid:
        uniform_row = [1.0 / expected_columns] * expected_columns
        matrix = [uniform_row[:] for _ in range(expected_rows)]
        print("Invalid distribution file. Using uniform distribution.")
    return matrix

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    train_and_save_model()
