import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import csv
from sb3_contrib import MaskablePPO
# Add root project path to sys.path if running from subdirectory like eval/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_env.elevatorGym import ElevatorEnv

# Configuration
NUM_RUNS = 20  # for quick testing; increase for full evaluation
SIM_TIME = 300
STEP_INTERVAL = 10
NUM_FLOORS = 10
NUM_ELEVATORS = 3
CAPACITY = 5
TRAVEL_TIME = 1.0
STOP_TIME = 0.5
ELEVATOR_CHECK_INTERVAL = 0.2
ARRIVAL_FILE = "../data/averageArivalTimes.txt"
FLOOR_FILE = "../data/floorDistribution.txt"
MODEL_PATH = "../model/ppo_multi_elevator"
RESULT_DIR = "results"
RL_CSV = os.path.join(RESULT_DIR, "rl_results.csv")
SC_CSV = os.path.join(RESULT_DIR, "scanning_results.csv")

# Helper: load arrival times per second from file (fallback to constant default)
def load_arrival_times(file_path, default=20.0, sim_time=300):
    try:
        with open(file_path, "r") as f:
            line = f.readline()
            values = list(map(float, line.split()))
            return values[:sim_time] if len(values) >= sim_time else [default] * sim_time
    except:
        return [default] * sim_time

# Helper: load floor destination probabilities per timestep from file
def load_floor_distribution(file_path, num_floors, sim_time):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            matrix = [list(map(float, line.split()[:num_floors])) for line in lines if line.strip()]
            return matrix[:sim_time] if len(matrix) >= sim_time else [[1.0 / num_floors] * num_floors] * sim_time
    except:
        return [[1.0 / num_floors] * num_floors for _ in range(sim_time)]

# Save metrics to CSV file for reuse
def save_to_csv(filename, data_dict, time_steps):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time"] + list(data_dict.keys()))
        for i, t in enumerate(time_steps):
            row = [t] + [data_dict[key][i] for key in data_dict]
            writer.writerow(row)

# Load previously saved metrics from CSV
def load_from_csv(filename):
    data = {"waiting": [], "traveling": [], "idle": [], "building": []}
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data

# Run a single simulation and collect metrics at defined time intervals
def run_simulation_timed(use_model):
    floor_dist = load_floor_distribution(FLOOR_FILE, NUM_FLOORS, SIM_TIME)
    arrival_times = load_arrival_times(ARRIVAL_FILE, 20.0, SIM_TIME)

    env = ElevatorEnv(
        floor_dist,
        arrival_times,
        num_floors=NUM_FLOORS,
        num_elevators=NUM_ELEVATORS,
        elevator_capacity=CAPACITY,
        simulation_time=SIM_TIME,
        fahrzeit=TRAVEL_TIME,
        halt_zeit=STOP_TIME,
        elevator_check_interval=ELEVATOR_CHECK_INTERVAL,
        scanning=not use_model
    )

    house = env.house
    house.gym_env = env

    # Use trained model or fallback to rule-based dispatcher
    if use_model:
        model = MaskablePPO.load(MODEL_PATH)
        house.model = model
        house.run_controller()
    else:
        house.run_dispatcher()

    # Store metrics at intervals
    waiting, traveling, idle, building = [], [], [], []

    for t in range(STEP_INTERVAL, SIM_TIME + 1, STEP_INTERVAL):
        house.env.run(until=t)
        metrics = house.get_average_metrics()
        waiting.append(metrics["average_waiting_time"])
        traveling.append(metrics["average_traveling_time"])
        idle.append(metrics["average_idle_time"])
        building.append(metrics["average_time_in_building"])

    return waiting, traveling, idle, building

# Run multiple simulations and average the collected metrics
def benchmark(use_model, filename):
    if os.path.exists(filename):
        print(f"Loading cached results from {filename} ...")
        return load_from_csv(filename)

    waiting_all, traveling_all, idle_all, building_all = [], [], [], []
    label = "RL Model" if use_model else "Scanning"
    print(f"Running simulations for: {label} ...")

    for _ in tqdm(range(NUM_RUNS)):
        w, tr, idl, bld = run_simulation_timed(use_model)
        waiting_all.append(w)
        traveling_all.append(tr)
        idle_all.append(idl)
        building_all.append(bld)

    result = {
        "waiting": np.mean(waiting_all, axis=0),
        "traveling": np.mean(traveling_all, axis=0),
        "idle": np.mean(idle_all, axis=0),
        "building": np.mean(building_all, axis=0),
    }

    time_steps = np.arange(STEP_INTERVAL, SIM_TIME + 1, STEP_INTERVAL)
    save_to_csv(filename, result, time_steps)
    return result

# Main execution block
if __name__ == "__main__":
    rl = benchmark(use_model=True, filename=RL_CSV)
    sc = benchmark(use_model=False, filename=SC_CSV)
    time_steps = np.arange(STEP_INTERVAL, SIM_TIME + 1, STEP_INTERVAL)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    titles = [
        "Average Waiting Time",
        "Average Traveling Time",
        "Average Idle Time",
        "Average Time in Building"
    ]
    keys = ["waiting", "traveling", "idle", "building"]

    for i in range(4):
        axs[i].plot(time_steps, rl[keys[i]], label="RL Model", linewidth=2)
        axs[i].plot(time_steps, sc[keys[i]], label="Scanning", linewidth=2)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Simulation Time")
        axs[i].set_ylabel("Time")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
