import logging
import os
import simpy
import random as rnd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .Person import Person
from .Floor import Floor
from .Aufzug import Aufzug
from .Aufzug_scanning import Aufzug_scanning
from .Dispatcher import Dispatcher
from .ElevatorController import ElevatorController

# Ensure log directory exists
os.makedirs("../logs", exist_ok=True)

# Logger setup
logger = logging.getLogger("house")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("../logs/house.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

class House:
    """
    Represents the entire building simulation.

    Manages:
      - Floors
      - Elevators
      - People spawning
      - Building navigation (dynamic reassignment)
      - Central dispatcher
    """

    # Simulation Constants
    AVERAGE_ARRIVAL_TIME = 20.0             # Default average time between new persons
    CHECK_INTERVAL_NAVIGATOR = 10.0         # Navigator check interval (reassign people)
    CHECK_INTERVAL_DISPATCHER = 0.2         # Dispatcher check interval (elevator dispatch)
    CHECK_INTERVAL_DESPAWNER = 0.1          # Despawner check interval (person despawner)
    LEAVE_PROBABILITY = 0.8                 # Chance that a person leaves when on the ground floor
    STAY_ON_GROUND_PROBABILITY = 0.2        # Chance that a person stays on the ground floor

    def __init__(self, floorDistribution, averageArrivalTimes, num_floors: int, num_elevators: int, elevator_capacity: int = 5, simulation_time: int = 200,
                 fahrzeit: float = 1.0, halt_zeit: float = 0.5, elevator_check_interval: float = 0.2, scanning=False):
        # Initialize SimPy environment
        self.env = simpy.Environment()
        self.simulation_time = simulation_time

        self.num_floors = num_floors

        # Create floors
        self.floors = [Floor(self.env, i) for i in range(num_floors)]

        # Create elevators (standard or scanning mode)
        if not scanning:
            self.elevators = []
            for eid in range(num_elevators):
                aufzug = Aufzug(
                    env=self.env,
                    aufzug_id=eid,
                    kapazitaet=elevator_capacity,
                    house=self,
                    fahrzeit=fahrzeit,
                    halte_zeit=halt_zeit,
                    check_interval_elevator=elevator_check_interval
                )
                self.elevators.append(aufzug)
            
        if scanning:
            self.elevators = []
            for eid in range(num_elevators):
                aufzug = Aufzug_scanning(
                    env=self.env,
                    aufzug_id=eid,
                    kapazitaet=elevator_capacity,
                    house=self,
                    fahrzeit=fahrzeit,
                    halte_zeit=halt_zeit,
                    check_interval_elevator=elevator_check_interval
                )
                self.elevators.append(aufzug)
                self.env.process(aufzug.run())

        # Manage people
        self.personList = []
        self.person_counter = 0

        # Metrics
        self.waiting_times = []
        self.time_in_building = []

        # Start background processes
        self.env.process(self.person_spawner())
        self.env.process(self.building_navigator())
        self.env.process(self.person_despawner())

        # Load external configurations
        self.floor_distribution = floorDistribution
        self.averageArrivalTimes = averageArrivalTimes 
    
    def run_controller(self):
        # Create and start the elevator controller
        self.controller = ElevatorController(self.env, self.model, self.gym_env)
        self.env.process(self.controller.elevator_controller(interval=self.CHECK_INTERVAL_DISPATCHER))
    
    def run_dispatcher(self):
        # Create and start the elevator dispatcher
        dispatcher = Dispatcher(self.env, self, check_interval=self.CHECK_INTERVAL_DISPATCHER)
        self.env.process(dispatcher.run())

    def person_spawner(self):
        """
        Spawns people at the ground floor (floor 0) according to dynamic arrival rates.
        With a certain probability they stay at ground floor idle,
        otherwise they get assigned a random target floor.
        """
        while True:
            time_step = int(self.env.now)

            # Get dynamic average arrival time
            avg_arrival = self.averageArrivalTimes[min(time_step, len(self.averageArrivalTimes) - 1)]

            # Draw next arrival time from exponential distribution
            warten = np.random.exponential(avg_arrival) if avg_arrival > 0 else 1e6
            yield self.env.timeout(warten)

            start = 0  # always floor 0

            # Create new person (target is temporary and may change later)
            p = Person(
                person_id=self.person_counter,
                startFloor=start,
                targetFloor=start,
                spawn_time=self.env.now
            )
            self.person_counter += 1
            self.personList.append(p)

            floor = self.floors[start]

            # Decide if the person stays idle at ground floor
            if rnd.random() < House.STAY_ON_GROUND_PROBABILITY:
                yield floor.idlePeople.put(p)
                logger.info(f"[t={self.env.now:.1f}] {p} spawns at ground floor and stays idle.")
                continue

            # Otherwise assign a random target floor
            probs = self.floor_distribution[min(time_step, len(self.floor_distribution) - 1)]
            target = rnd.choices(
                population=list(range(1, self.num_floors)),
                weights=probs[1:self.num_floors],
                k=1
            )[0]
            p.targetFloor = target

            # Person enters the elevator queue
            p.enter_queue_time = self.env.now
            yield floor.elevatorQueueUp.put(p)
            logger.info(f"[t={self.env.now:.1f}] NEW {p} spawns at ground floor, target={target}")
    
    def person_despawner(self):
        """
        Periodically checks for idle people at ground floor (floor 0).
        They may leave the building with a certain leave probability.
        """
        while True:
            yield self.env.timeout(House.CHECK_INTERVAL_DESPAWNER)

            ground_floor = self.floors[0]
            if not ground_floor.idlePeople:
                continue

            people_to_remove = []

            for person in list(ground_floor.idlePeople.items):
                if rnd.random() < House.LEAVE_PROBABILITY:
                    person.leave_building_time = self.env.now
                    self.time_in_building.append(person.leave_building_time - person.spawn_time)
                    people_to_remove.append(person)
                    logger.info(f"[t={self.env.now:.1f}] {person} was idle and leaves the building.")

            for p in people_to_remove:
                ground_floor.idlePeople.items.remove(p)
                self.personList.remove(p)

    def building_navigator(self):
        """
        Periodically reassigns one random idle person to a new target floor.
        """
        while True:
            yield self.env.timeout(House.CHECK_INTERVAL_NAVIGATOR)

            time_index = int(self.env.now)
            if time_index >= len(self.floor_distribution):
                time_index = len(self.floor_distribution) - 1

            # Collect idle people from all floors
            all_idle_people = []
            for floor in self.floors:
                all_idle_people.extend(floor.idlePeople.items)

            if not all_idle_people:
                continue

            # Pick a random idle person
            person = rnd.choice(all_idle_people)
            current_floor = person.currentFloor
            floor = self.floors[current_floor]

            # Assign a new target floor
            probs = self.floor_distribution[time_index]
            new_target = rnd.choices(
                population=list(range(0, self.num_floors)),
                weights=probs[0:self.num_floors],
                k=1
            )[0]

            # Avoid choosing the same floor as current
            while new_target == current_floor:
                new_target = rnd.choices(
                    population=list(range(0, self.num_floors)),
                    weights=probs[0:self.num_floors],
                    k=1
                )[0]
            
            # Update person state
            person.targetFloor = new_target
            person.enter_queue_time = self.env.now
            floor.idlePeople.items.remove(person)

            if new_target < current_floor:
                yield floor.elevatorQueueDown.put(person)
                logger.info(f"[t={self.env.now:.1f}] {person} on {floor} chooses new target {new_target} (DOWN).")
            else:
                yield floor.elevatorQueueUp.put(person)
                logger.info(f"[t={self.env.now:.1f}] {person} on {floor} chooses new target {new_target} (UP).")

    def get_average_metrics(self):
        """
        Computes and returns average statistics:
          - Waiting time
          - Traveling time
          - Elevator idle time
          - Total building stay time
        """
        avg_waiting_time = np.mean(self.waiting_times) if self.waiting_times else 0
        avg_building_time = np.mean(self.time_in_building) if self.time_in_building else 0

        all_traveling_times = []
        all_idle_times = []

        for elev in self.elevators:
            all_traveling_times.extend(elev.traveling_times)
            all_idle_times.extend(elev.idle_times)

        avg_traveling_time = np.mean(all_traveling_times) if all_traveling_times else 0
        avg_idle_time = np.mean(all_idle_times) if all_idle_times else 0

        return {
            "average_waiting_time": avg_waiting_time,
            "average_traveling_time": avg_traveling_time,
            "average_idle_time": avg_idle_time,
            "average_time_in_building": avg_building_time
        }
    
    def reset_metrics(self):
        """
        Resets collected metrics for a fresh simulation run.
        """
        self.waiting_times = []
        self.time_in_building = []
        for elev in self.elevators:
            elev.traveling_times = []
            elev.idle_times = []
    
    def get_state(self):
        """
        Returns the current state of the house (for use in external components or models).
        """
        return {
            'floors': self.floors,
            'simpy_env': self.env,
            'elevators': self.elevators
        }
