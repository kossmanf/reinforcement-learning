import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from elevatorGym import ElevatorEnv 
import time
from stable_baselines3.common.callbacks import BaseCallback

def make_env():
    """
    Factory function to create a new instance of the environment.
    """
    return ElevatorEnv(
        sim_controller=None,
        num_floors=5,
        num_elevators=2,
        elevator_capacity=4,
        simulation_time=200,
        fahrzeit=1.0,
        halt_zeit=0.5,
        elevator_check_interval=1.0,
        arrival_file="averageArivalTimes.txt",
        floor_file="floorDistribution.txt"
    )

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"⏳ Training started: {self.total_timesteps} total timesteps")

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            print(f"Step {self.num_timesteps}/{self.total_timesteps} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        return True

def train_and_save_model():
    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1)

    total_timesteps = 100_000
    callback = ProgressBarCallback(total_timesteps=total_timesteps, print_freq=1000)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("ppo_elevator")

    print("Model training complete and saved as 'ppo_multi_elevator.zip")


def load_and_run_model():
    """
    Load and run the model in the environment.
    """
    model = PPO.load("ppo_multi_elevator")
    env = make_env()
    obs, info = env.reset()
    done = False

    print("Running trained model...\n")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")


if __name__ == "__main__":
    train_and_save_model()
    load_and_run_model()