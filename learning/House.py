import logging
import simpy
import random as rnd
import numpy as np

from Person import Person
from Floor import Floor
from Aufzug import Aufzug
#  from Dispatcher import Dispatcher

# Logger setup
logger = logging.getLogger(__name__)

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
                 fahrzeit: float = 1.0, halt_zeit: float = 0.5, elevator_check_interval: float = 0.2,
                 arrival_file: str = "averageArivalTimes.txt", floor_file: str = "floorDistribution.txt"):
        # Initialize SimPy environment
        self.env = simpy.Environment()
        self.simulation_time = simulation_time

        self.num_floors = num_floors

        # Create floors
        self.floors = [Floor(self.env, i) for i in range(num_floors)]

        # Create elevators
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
            #self.env.process(aufzug.run())

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

        # Dispatcher
        # self.dispatcher = Dispatcher(self.env, self, check_interval=self.CHECK_INTERVAL_DISPATCHER)
        # self.env.process(self.dispatcher.run())

        # Load external configurations
        #self.loadFloorDistribution(floor_file)
        #self.loadAverageArivalTimes(arrival_file)
        self.floor_distribution = floorDistribution
        self.averageArrivalTimes = averageArrivalTimes 

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

            # Create new person
            p = Person(
                person_id=self.person_counter,
                startFloor=start,
                targetFloor=start,  # erstmal temporär, eventuell wird Ziel geändert
                spawn_time=self.env.now
            )
            self.person_counter += 1
            self.personList.append(p)

            floor = self.floors[start]

            # Decide if the person stays at ground floor idle
            if rnd.random() < House.STAY_ON_GROUND_PROBABILITY:
                yield floor.idlePeople.put(p)
                logger.info(f"[t={self.env.now:.1f}] {p} spawnt im EG und bleibt dort idle.")
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
            logger.info(f"[t={self.env.now:.1f}] NEUE {p} spawnt im EG, Ziel={target}")
    
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
                    logger.info(f"[t={self.env.now:.1f}] {person} wartet im EG und verlässt das Gebäude.")

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

            # Collect idle people
            all_idle_people = []
            for floor in self.floors:
                all_idle_people.extend(floor.idlePeople.items)

            if not all_idle_people:
                continue  # No idle person to reassign

            # Pick a random idle person
            person = rnd.choice(all_idle_people)
            current_floor = person.currentFloor
            floor = self.floors[current_floor]

            # Otherwise assign a new target floor
            probs = self.floor_distribution[time_index]
            new_target = rnd.choices(
                population=list(range(1, self.num_floors)),
                weights=probs[1:self.num_floors],
                k=1
            )[0]

            # Make sure person does not choose their current floor
            while new_target == current_floor:
                new_target = rnd.choices(
                    population=list(range(1, self.num_floors)),
                    weights=probs[1:self.num_floors],
                    k=1
                )[0]

            # Update person state
            person.targetFloor = new_target
            person.enter_queue_time = self.env.now
            floor.idlePeople.items.remove(person)

            if new_target > current_floor:
                yield floor.elevatorQueueDown.put(person)
                logger.info(f"[t={self.env.now:.1f}] {person} auf {floor} wählt neues Ziel {new_target} (UP).")
            else:
                yield floor.elevatorQueueUp.put(person)
                logger.info(f"[t={self.env.now:.1f}] {person} auf {floor} wählt neues Ziel {new_target} (DOWN).")

    def loadFloorDistribution(self, fileName):
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

        expected_columns = len(self.floors)
        expected_rows = self.simulation_time

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

        self.floor_distribution = matrix

    def loadAverageArivalTimes(self, fileName):
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
            matrix = [self.AVERAGE_ARRIVAL_TIME] * self.simulation_time

        expected_length = self.simulation_time
        if len(matrix) < expected_length:
            print(f"Invalid data length. Expected {expected_length}, got {len(matrix)}.")
            matrix = [self.AVERAGE_ARRIVAL_TIME] * expected_length

        self.averageArrivalTimes = matrix

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
        self.waiting_times = []
        self.time_in_building = []
        for elev in self.elevators:
            elev.traveling_times = []
            elev.idle_times = []
    
    def get_state(self):
        return {
            'floors': self.floors,
            'simpy_env': self.env,
            'elevators': self.elevators
        }
