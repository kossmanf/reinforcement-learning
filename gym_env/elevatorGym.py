import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.House import House
import logging

# Ensure the log directory exists
os.makedirs("../logs", exist_ok=True)

# Logger setup
logger = logging.getLogger("elevator_env")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("../logs/elevator_env.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)


class ElevatorEnv(gym.Env):
    """
    Gymnasium environment for controlling multiple elevators via reinforcement learning.
    """

    def __init__(self, floorDistribution, averageArrivalTimes, num_floors: int, num_elevators: int, elevator_capacity: int = 5, simulation_time: int = 200,
                 fahrzeit: float = 1.0, halt_zeit: float = 0.5, elevator_check_interval: float = 0.2, scanning=False):
        super().__init__()

        # Store parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity
        self.simulation_time = simulation_time
        self.fahrzeit = fahrzeit
        self.halt_zeit = halt_zeit
        self.elevator_check_interval = elevator_check_interval
        self.averageArrivalTimes = averageArrivalTimes
        self.floorDistribution = floorDistribution
        self.scanning = scanning

        self.num_training_floors = 10

        # Create simulation environment (House)
        self.house = House(
            self.floorDistribution, self.averageArrivalTimes, self.num_training_floors, self.num_elevators,
            self.elevator_capacity, self.simulation_time, self.fahrzeit, self.halt_zeit,
            self.elevator_check_interval, scanning
        )

        # Action space: 0=up, 1=down, 2=stop for each elevator
        self.num_actions_per_elevator = 3
        self.action_space = spaces.Discrete(self.num_actions_per_elevator * self.num_elevators)

        # Observation space setup:
        # Elevator state: [floor (one-hot), direction (3-bit), passenger count (one-hot), target floors (binary)]
        # Waiting queues: [up/down flags per floor]
        floor_low = [0] * self.num_floors
        direction_low = [0, 0, 0]
        passenger_low = [0] * self.elevator_capacity
        targets_low = [0] * self.num_floors
        elevator_low = floor_low + direction_low + passenger_low + targets_low
        waiting_low = [0] * self.num_floors * 2

        floor_high = [1] * self.num_floors
        direction_high = [1, 1, 1]
        passenger_high = [1] * self.elevator_capacity
        targets_high = [1] * self.num_floors
        elevator_high = floor_high + direction_high + passenger_high + targets_high
        waiting_high = [1] * self.num_floors * 2

        low = np.array(elevator_low * self.num_elevators + waiting_low)
        high = np.array(elevator_high * self.num_elevators + waiting_high)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

    def reset(self, seed=None, options=None):
        # Reset the simulation environment and return initial observation
        self.house = House(
            self.floorDistribution, self.averageArrivalTimes, self.num_training_floors, self.num_elevators,
            self.elevator_capacity, self.simulation_time, self.fahrzeit, self.halt_zeit,
            self.elevator_check_interval, self.scanning
        )
        house_state = self.get_house_state()
        elev_states = self.get_elev_states()
        info = {"action_mask": self.getMask()}
        return self.get_observation(house_state, elev_states), info

    def step(self, action_index):
        """
        Executes one action affecting a single elevator (up/down/stop).
        """
        # Decode action index
        elevator_index = action_index // self.num_actions_per_elevator
        action_id = action_index % self.num_actions_per_elevator
        action_str = {0: "up", 1: "down", 2: "stop"}[action_id]
        elevator = self.house.elevators[elevator_index]

        # Schedule and run the action
        self.house.env.process(elevator.execute_action(action_str))
        self.house.env.run(until=self.house.env.now + 1)

        # Compute reward
        rewards = [e.last_reward for e in self.house.elevators]
        total_reward = np.clip(sum(rewards), 0, 13)
        normalized_reward = total_reward / 13

        # Build new observation
        house_state = self.get_house_state()
        elev_states = self.get_elev_states()
        observation = self.get_observation(house_state, elev_states)

        # Check termination condition
        done = self.house.env.now >= self.simulation_time

        # Log step details
        log_msg = f"\n[STEP] Time: {self.house.env.now}\n" \
                  f"Action taken: Elevator {elevator_index} -> {action_str} (Index: {action_index})\n"
        for i, elev_state in enumerate(elev_states):
            log_msg += f"Elevator {i}: {elev_state}\n"
        logger.info(log_msg)

        info = {"action_mask": self.getMask()}
        return observation, normalized_reward, done, False, info

    def get_observation(self, house_state, elev_states):
        # Encodes elevator and floor state into a single observation vector
        obs = []

        for elev_state in elev_states:
            # Encode current floor as one-hot
            obs.extend([1 if f == elev_state["floor"] else 0 for f in range(self.num_floors)])

            # Encode direction as 3-bit
            direction_encoding = {1: [1, 1, 0], 0: [0, 1, 0], -1: [0, 0, 1]}
            obs.extend(direction_encoding.get(elev_state["direction"], [0, 0, 0]))

            # Encode passenger count as binary vector
            obs.extend([1 if i < elev_state["passenger_count"] else 0 for i in range(self.elevator_capacity)])

            # Encode target floors
            targets = elev_state.get("zielstoecke", [])
            obs.extend([1 if f in targets else 0 for f in range(self.num_floors)])

        for floor in house_state['floors']:
            obs.append(1 if len(floor.elevatorQueueUp.items) > 0 else 0)
            obs.append(1 if len(floor.elevatorQueueDown.items) > 0 else 0)

        # Fill up remaining floor slots if num_training_floors < num_floors
        for _ in range(0, (self.num_floors - self.num_training_floors)):
            obs.append(0)
            obs.append(0)

        return np.array(obs, dtype=np.int32)

    def get_elev_states(self):
        return [e.get_state() for e in self.house.elevators]

    def get_house_state(self):
        return self.house.get_state()

    def decode_observation(self, obs, num_elevators, num_floors):
        """
        Decodes and prints the observation vector for debugging.
        """
        index = 0
        print("\n--- Elevator States ---")
        for i in range(num_elevators):
            floor_vector = obs[index:index + num_floors]
            floor = np.argmax(floor_vector)
            index += num_floors

            dir_vector = tuple(obs[index:index + 3])
            direction_lookup = {(1, 1, 0): 1, (0, 1, 0): 0, (0, 0, 1): -1}
            direction = direction_lookup.get(dir_vector, "Unknown")
            index += 3

            passenger_bits = obs[index:index + self.elevator_capacity]
            passenger_count = sum(passenger_bits)
            index += self.elevator_capacity

            print(f"Elevator {i}: Floor={floor}, Direction={direction}, Passengers={passenger_count}")

        print("\n--- Elevator Target Floors ---")
        for i in range(num_elevators):
            targets = obs[index:index + num_floors]
            target_floors = [f for f, active in enumerate(targets) if active == 1]
            print(f"Elevator {i}: Targets={target_floors}")
            index += num_floors

        print("\n--- Waiting People per Floor ---")
        for floor in range(num_floors):
            up_waiting = obs[index]
            down_waiting = obs[index + 1]
            print(f"Floor {floor}: Waiting Up={bool(up_waiting)}, Waiting Down={bool(down_waiting)}")
            index += 2

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def getMask(self):
        return np.array([self.getElevatorActionMask(e) for e in self.house.elevators], dtype=bool)

    def getElevatorActionMask(self, elevator):
        # Determines valid actions for one elevator
        mask = np.zeros(3, dtype=bool)
        mask[0] = True  # up
        mask[1] = True  # down
        mask[2] = False  # stop

        current_floor = elevator.current_floor
        floor = self.house.floors[current_floor]

        exiting = any(p.targetFloor == current_floor for p in elevator.passengers)
        entering = len(floor.elevatorQueueUp.items) > 0 or len(floor.elevatorQueueDown.items) > 0

        queue = floor.elevatorQueueUp if elevator.direction != -1 else floor.elevatorQueueDown
        entering_direction = len(queue.items) > 0

        empty = elevator.is_empty()
        waiting = elevator.is_waiting()
        full = len(elevator.passengers) == elevator.kapazitaet

        # Restrict up at top floor
        if current_floor == self.num_training_floors - 1:
            mask[0] = False

        # Restrict down at bottom floor
        if current_floor == 0:
            mask[1] = False

        # If entering/exiting is possible, force stop
        if exiting or (entering_direction and not full):
            mask[0] = False
            mask[1] = False
            mask[2] = True

        # If empty and no one waiting, force stop
        if not waiting and empty:
            mask[0] = False
            mask[1] = False
            mask[2] = True

        return mask

    def get_metrics(self):
        if hasattr(self, "house") and hasattr(self.house, "get_average_metrics"):
            return self.house.get_average_metrics()
        return {}
