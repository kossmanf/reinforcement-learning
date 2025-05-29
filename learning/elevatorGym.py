import gymnasium as gym
from gymnasium import spaces
import numpy as np
from House import House

class ElevatorEnv(gym.Env):
    """
    Gymnasium environment for controlling multiple elevators via reinforcement learning.
    """

    def __init__(self, floorDistribution, averageArrivalTimes, num_floors: int, num_elevators: int, elevator_capacity: int = 5, simulation_time: int = 200,
                 fahrzeit: float = 1.0, halt_zeit: float = 0.5, elevator_check_interval: float = 0.2):
        super().__init__()

        #  setting the parameters parameters
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity
        self.simulation_time = simulation_time
        self.fahrzeit = fahrzeit
        self.halt_zeit = halt_zeit
        self.elevator_check_interval = elevator_check_interval
        self.averageArrivalTimes = averageArrivalTimes
        self.floorDistribution = floorDistribution

        # creating the house
        self.house = House(self.floorDistribution, self.averageArrivalTimes, self.num_floors, self.num_elevators, self.elevator_capacity, self.simulation_time,
                self.fahrzeit, self.halt_zeit, self.elevator_check_interval,'','')

        # Action space: for each elevator one action (0=up, 1=down, 2=stop)
        self.num_actions_per_elevator = 3  # up, down, stop
        self.action_space = spaces.Discrete(self.num_actions_per_elevator * self.num_elevators)

        # observation 
        # low: [0, -1, 0] for every elevator (floor, dircection, passenger_count)
        # number of people waiting on each floor up or down up to 1000 people per floor possible

        # Observation space: state of each elevator + waiting queues
        low = np.array([0, -1, 0] * self.num_elevators + [0] * self.num_floors * 2)

        high = np.array(
            [self.num_floors - 1, 1, elevator_capacity] * self.num_elevators + [1000] * self.num_floors * 2
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)


    def reset(self, seed=None, options=None):
        self.house =  House(self.floorDistribution, self.averageArrivalTimes, self.num_floors, self.num_elevators, self.elevator_capacity, self.simulation_time,
                self.fahrzeit, self.halt_zeit, self.elevator_check_interval,'','')
                
        house_state = self.get_house_state()
        elev_states = self.get_elev_states()

        return self.get_observation(house_state, elev_states), {}

    def step(self, action_index):
        """
        Executes one discrete action that affects one elevator (up/down/stop).
        """
        # Inline decoding of action
        elevator_index = action_index // self.num_actions_per_elevator
        action_id = action_index % self.num_actions_per_elevator
        action_str = {0: "up", 1: "down", 2: "stop"}[action_id]

        # Trigger elevator action
        self.house.env.process(self.house.elevators[elevator_index].execute_action(action_str))

        # Advance simulation
        self.house.env.run(until=self.house.env.now + 1)

        # Reward collection + normalization
        rewards = [elevator.last_reward for elevator in self.house.elevators]
        total_reward = np.clip(sum(rewards), 0, 500)
        normalized_reward = total_reward / 500.0

        # Build new observation
        house_state = self.get_house_state()
        elev_states = self.get_elev_states()
        observation = self.get_observation(house_state, elev_states)

        # End condition
        done = self.house.env.now >= self.simulation_time

        return observation, normalized_reward, done, False, {}

    
    def get_observation(self, house_state, elev_states):
        obs = []

        for elev_state in elev_states:
            obs.extend([
                elev_state["floor"],
                elev_state["direction"],
                elev_state["passenger_count"]
            ])
        
        hous_floors = house_state['floors']
        for floor in hous_floors:
            obs.append(len(floor.elevatorQueueUp.items))
            obs.append(len(floor.elevatorQueueDown.items))
        
        return np.array(obs, dtype=np.int32)
    
    def get_elev_states(self):
        elev_states = []
        
        for elev in self.house.elevators:
            elev_states.append(elev.get_state())

        return elev_states
    
    def get_house_state(self):
        return self.house.get_state()