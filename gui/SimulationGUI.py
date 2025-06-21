import tkinter as tk
import threading
import matplotlib.pyplot as plt
import logging
import matplotlib.image as mpimg
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import traceback
from env.House import House
from gui.Simulation_plot import draw_simulation_step, plot_metrics_over_time
from gym_env.elevatorGym import ElevatorEnv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.House import House

class SimulationGUI:
    def __init__(self, root):
        # Initialize the main window and input fields
        self.root = root
        self.root.title("Elevator Simulation Parameters")

       # Section title: Floor and Elevator Settings
        tk.Label(root, text="Floor and Elevator Settings", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(10,2))

        tk.Label(root, text="Number of Floors:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.floors_var = tk.StringVar(value="10")
        tk.Entry(root, textvariable=self.floors_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(root, text="Number of Elevators:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.elevators_var = tk.StringVar(value="3")
        tk.Entry(root, textvariable=self.elevators_var).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(root, text="Elevator Capacity:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.capacity_var = tk.StringVar(value="5")
        tk.Entry(root, textvariable=self.capacity_var).grid(row=3, column=1, padx=5, pady=5)

        # Section title: Simulation Settings
        tk.Label(root, text="Simulation Settings", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=(10,2))

        tk.Label(root, text="Total Simulation Time:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.total_time_var = tk.StringVar(value="300")
        tk.Entry(root, textvariable=self.total_time_var).grid(row=5, column=1, padx=5, pady=5)

        tk.Label(root, text="Step Size:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.step_size_var = tk.StringVar(value="1")
        tk.Entry(root, textvariable=self.step_size_var).grid(row=6, column=1, padx=5, pady=5)

        # Section title: Extra Adjustable Parameters
        tk.Label(root, text="Extra Adjustable Parameters", font=("Arial", 10, "bold")).grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=(10,2))

        tk.Label(root, text="Average Arrival Time:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.arrival_time_var = tk.StringVar(value="20.0")
        tk.Entry(root, textvariable=self.arrival_time_var).grid(row=8, column=1, padx=5, pady=5)

        tk.Label(root, text="Leave Probability (0-1):").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.leave_prob_var = tk.StringVar(value="0.8")
        tk.Entry(root, textvariable=self.leave_prob_var).grid(row=9, column=1, padx=5, pady=5)

        tk.Label(root, text="Stay on Ground Probability (0-1):").grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
        self.stay_on_ground_prob_var = tk.StringVar(value="0.2")
        tk.Entry(root, textvariable=self.stay_on_ground_prob_var).grid(row=10, column=1, padx=5, pady=5)

        tk.Label(root, text="Elevator Travel Time per Floor:").grid(row=11, column=0, sticky=tk.W, padx=5, pady=5)
        self.travel_time_var = tk.StringVar(value="1.0")
        tk.Entry(root, textvariable=self.travel_time_var).grid(row=11, column=1, padx=5, pady=5)

        tk.Label(root, text="Elevator Stop Time at Floor:").grid(row=12, column=0, sticky=tk.W, padx=5, pady=5)
        self.stop_time_var = tk.StringVar(value="0.5")
        tk.Entry(root, textvariable=self.stop_time_var).grid(row=12, column=1, padx=5, pady=5)

        tk.Label(root, text="Dispatcher Check Interval:").grid(row=13, column=0, sticky=tk.W, padx=5, pady=5)
        self.dispatcher_interval_var = tk.StringVar(value="0.2")
        tk.Entry(root, textvariable=self.dispatcher_interval_var).grid(row=13, column=1, padx=5, pady=5)

        tk.Label(root, text="Navigator Check Interval:").grid(row=14, column=0, sticky=tk.W, padx=5, pady=5)
        self.navigator_interval_var = tk.StringVar(value="10.0")
        tk.Entry(root, textvariable=self.navigator_interval_var).grid(row=14, column=1, padx=5, pady=5)

        tk.Label(root, text="Elevator Check Interval:").grid(row=15, column=0, sticky=tk.W, padx=5, pady=5)
        self.elevator_interval_var = tk.StringVar(value="0.2")
        tk.Entry(root, textvariable=self.elevator_interval_var).grid(row=15, column=1, padx=5, pady=5)

        tk.Label(root, text="Font Size:").grid(row=16, column=0, sticky=tk.W, padx=5, pady=5)
        self.fontsize_var = tk.StringVar(value="5")
        tk.Entry(root, textvariable=self.fontsize_var).grid(row=16, column=1, padx=5, pady=5)

        # Section title: File Paths
        tk.Label(root, text="File Paths", font=("Arial", 10, "bold")).grid(row=17, column=0, columnspan=2, sticky="w", padx=5, pady=(10,2))

        tk.Label(root, text="Arrival Times File Path:").grid(row=18, column=0, sticky=tk.W, padx=5, pady=5)
        self.arrival_file_var = tk.StringVar(value="data/averageArivalTimes.txt")
        tk.Entry(root, textvariable=self.arrival_file_var).grid(row=18, column=1, padx=5, pady=5)

        tk.Label(root, text="Floor Distribution File Path:").grid(row=19, column=0, sticky=tk.W, padx=5, pady=5)
        self.floor_file_var = tk.StringVar(value="data/floorDistribution.txt")
        tk.Entry(root, textvariable=self.floor_file_var).grid(row=19, column=1, padx=5, pady=5)

        tk.Label(root, text="Model File Path:").grid(row=20, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_file_var = tk.StringVar(value="model/ppo_multi_elevator")
        tk.Entry(root, textvariable=self.model_file_var).grid(row=20, column=1, padx=5, pady=5)

        # Checkbox to use RL model
        self.use_model_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Use Reinforcement Learning Model", variable=self.use_model_var).grid(row=21, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

        # Status label and buttons
        self.status_label = tk.Label(root, text="Ready", fg="blue")
        self.status_label.grid(row=22, column=0, columnspan=2, pady=5)

        self.start_button = tk.Button(root, text="Start Simulation", command=self.on_start)
        self.start_button.grid(row=23, column=0, padx=5, pady=5)

        self.stop_button = tk.Button(root, text="Stop Simulation", command=self.on_stop, state=tk.DISABLED)
        self.stop_button.grid(row=23, column=1, padx=5, pady=5)

        # Simulation internal states
        self._stop_flag = False
        self.sim_thread = None
        self.fig = None

        self.house = None
        self.total_time = 0
        self.step_size = 0
        self.bg_image = None

    def on_start(self):
        # Read and validate user input values
        try:
            self.num_floors = int(self.floors_var.get())
            self.num_elevators = int(self.elevators_var.get())
            self.capacity = int(self.capacity_var.get())
            self.total_time = int(self.total_time_var.get())
            self.step_size = int(self.step_size_var.get())
            self.average_arrival_time = float(self.arrival_time_var.get())
            self.leave_probability = float(self.leave_prob_var.get())
            self.stay_on_ground_probability = float(self.stay_on_ground_prob_var.get())  # << HIER NEU
            self.travel_time = float(self.travel_time_var.get())
            self.stop_time = float(self.stop_time_var.get())
            self.dispatcher_interval = float(self.dispatcher_interval_var.get())
            self.navigator_interval = float(self.navigator_interval_var.get())
            self.elevator_interval = float(self.elevator_interval_var.get())
            self.arrival_file = self.arrival_file_var.get()
            self.floor_file = self.floor_file_var.get()
        except ValueError:
            self.status_label.config(text="Invalid input! Please enter valid numbers.", fg="red")
            return

        # Configure global House parameters
        House.AVERAGE_ARRIVAL_TIME = self.average_arrival_time
        House.LEAVE_PROBABILITY = self.leave_probability
        House.STAY_ON_GROUND_PROBABILITY = self.stay_on_ground_probability
        House.CHECK_INTERVAL_DISPATCHER = self.dispatcher_interval
        House.CHECK_INTERVAL_NAVIGATOR = self.navigator_interval
        
        plt.close('all')
        logging.info("Starting simulation...")

        # Initialize simulation environment (House instance)
        try:
            '''
            self.house = House(
                num_floors=self.num_floors,
                num_elevators=self.num_elevators,
                elevator_capacity=self.capacity,
                simulation_time=self.total_time,
                fahrzeit=self.travel_time,
                halt_zeit=self.stop_time,
                elevator_check_interval=self.elevator_interval,
                arrival_file=self.arrival_file,
                floor_file=self.floor_file
            )
            '''
            print(os.getcwd())
            self.run_simpy_with_model("model/ppo_multi_elevator")


        except Exception as e:
                self.status_label.config(text=f"Error: {e}", fg="red")
                # Print full traceback including the line where the error occurred
                tb_str = traceback.format_exc()
                print(tb_str)  # Or log it
                logging.error(f"Failed to initialize house: {e}\n{tb_str}")       
                return

        # Load background image if available
        self.bg_image = self.load_background_image()

        # Setup matplotlib figure with four subplots
        plt.ion()
        self.fig = plt.figure()
        self.fig.add_axes([0.05, 0.55, 0.4, 0.4])
        self.fig.add_axes([0.55, 0.55, 0.4, 0.4])
        self.fig.add_axes([0.05, 0.05, 0.4, 0.4])
        self.fig.add_axes([0.55, 0.05, 0.4, 0.4])
        self.fig.show()

        # Start simulation in a separate thread
        self._stop_flag = False
        self.status_label.config(text="Simulation running...", fg="green")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        def worker():
            # Continuously update the simulation visualization
            try:
                self.root.after(1, self.on_update)
            except Exception as e:
                logging.exception("Simulation failed")

        self.sim_thread = threading.Thread(target=worker, daemon=True)
        self.sim_thread.start()

    def load_background_image(self):
        # Load background image from file if available
        try:
            return mpimg.imread("gui/background.png")
        except FileNotFoundError:
            return None

    def advance_simulation_step(self, house, until_time):
        # Run the simulation environment until the specified time
        house.env.run(until=until_time)

    def on_update(self):
        # Update the simulation in steps and refresh visualization
        if self._stop_flag or self.house.env.now >= self.total_time:
            self.on_simulation_finished()
            return

        target_time = self.house.env.now + self.step_size
        self.advance_simulation_step(self.house, target_time)
        self.sample_averages_over_time(target_time)
        fontsize = int(self.fontsize_var.get())
        draw_simulation_step(self.fig, self.house, self.house.env.now, self.bg_image, fontsize)
        self.root.after(100, self.on_update)

    def on_stop(self):
        # Set flag to stop the simulation
        self.status_label.config(text="Stopping simulation...", fg="orange")
        self._stop_flag = True

    def on_simulation_finished(self):
        # Finalize the simulation and display results
        self.status_label.config(text="Simulation finished/stopped.", fg="blue")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if hasattr(self, "metrics_over_time"):
            fig = plt.figure(figsize=(10, 6))
            plot_metrics_over_time(self.metrics_over_time, fig)
        
        # resetting the calculated metrics
        self.house.reset_metrics()

        # Ich habe den Fehler behoben, dass die Listen nicht zwischen den Simulationsdurchläufen geleert wurden.
        # Dadurch wurden alte Werte weiter angehängt, was zu fehlerhaften Plots geführt hat.
        # Jetzt werden die Metriken vor jedem Durchlauf korrekt zurückgesetzt.
        self.metrics_over_time["time"] = []
        self.metrics_over_time["avg_waiting"] = []
        self.metrics_over_time["avg_traveling"] = []
        self.metrics_over_time["avg_idle"] = []
        self.metrics_over_time["avg_building"] = []


    def sample_averages_over_time(self, target_time):
        # Sample average performance metrics over time for plotting
        if not hasattr(self, "metrics_over_time"):
            self.metrics_over_time = {
                "time": [],
                "avg_waiting": [],
                "avg_traveling": [],
                "avg_idle": [],
                "avg_building": [],
            }

        averages = self.house.get_average_metrics()
        self.metrics_over_time["time"].append(target_time)
        self.metrics_over_time["avg_waiting"].append(averages["average_waiting_time"])
        self.metrics_over_time["avg_traveling"].append(averages["average_traveling_time"])
        self.metrics_over_time["avg_idle"].append(averages["average_idle_time"])
        self.metrics_over_time["avg_building"].append(averages["average_time_in_building"])

    def loadFloorDistribution(self, fileName, numFloors, simTime):
        try:
            with open(fileName, "r") as file:
                # Only take the first numFloors elements in each row
                matrix = [list(map(float, line.split()[:numFloors])) for line in file if line.strip()]
        except FileNotFoundError:
            print("File not found. Using uniform distribution.")
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

    def loadAverageArrivalTimes(self, fileName, default_time, simTime):
        try:
            with open(fileName, "r") as file:
                line = file.readline()
                matrix = list(map(float, line.split()))
        except FileNotFoundError:
            print("File not found. Using default average arrival times.")
            matrix = [default_time] * simTime

        if len(matrix) < simTime:
            print(f"Invalid data length. Expected {simTime}, got {len(matrix)}.")
            matrix = [default_time] * simTime

        return matrix
    
    def run_simpy_with_model(self, model_filename):
        # Load data
        floorDistribution = self.loadFloorDistribution(self.floor_file, 10, 300)
        averrageArivalTimes = self.loadAverageArrivalTimes(self.arrival_file, 20.0, 300)

        # Create Gym/SimPy environment
        self.gym_env = ElevatorEnv(
            floorDistribution,
            averrageArivalTimes,
            num_floors=self.num_floors,
            num_elevators=self.num_elevators,
            elevator_capacity=self.capacity,
            simulation_time=self.total_time,
            fahrzeit=self.travel_time,
            halt_zeit=self.stop_time,
            elevator_check_interval=self.elevator_interval,
            scanning=False
        )

        if self.use_model_var.get():
            # Create Gym/SimPy environment
            self.gym_env = ElevatorEnv(
                floorDistribution,
                averrageArivalTimes,
                num_floors=self.num_floors,
                num_elevators=self.num_elevators,
                elevator_capacity=self.capacity,
                simulation_time=self.total_time,
                fahrzeit=self.travel_time,
                halt_zeit=self.stop_time,
                elevator_check_interval=self.elevator_interval,
                scanning=False
            )

            # Load trained model
            self.model = MaskablePPO.load(model_filename)
            self.house =  self.gym_env.house
            self.house.model = self.model
            self.house.gym_env = self.gym_env
            self.house.run_controller()
        
        if not self.use_model_var.get():
            # Create Gym/SimPy environment
            self.gym_env = ElevatorEnv(
                floorDistribution,
                averrageArivalTimes,
                num_floors=self.num_floors,
                num_elevators=self.num_elevators,
                elevator_capacity=self.capacity,
                simulation_time=self.total_time,
                fahrzeit=self.travel_time,
                halt_zeit=self.stop_time,
                elevator_check_interval=self.elevator_interval,
                scanning=True
            )

            self.house =  self.gym_env.house
            self.house.gym_env = self.gym_env
            self.house.run_dispatcher()
