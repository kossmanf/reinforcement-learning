import logging
import simpy

# Logger setup
logger = logging.getLogger(__name__)

class Aufzug:
    """
    Represents an elevator using the SCAN algorithm ("elevator scan").

    Behavior:
      - Continues moving in the current direction as long as requests exist.
      - Changes direction only when necessary.
    
    Attributes:
        env (simpy.Environment): Simulation environment
        id (int): Elevator ID
        kapazitaet (int): Maximum passenger capacity
        akt_stock (int): Current floor
        richtung (int): Travel direction (-1=down, 0=idle, 1=up)
        fahrzeit (float): Travel time between adjacent floors
        halte_zeit (float): Waiting time at floors
        passagiere (list): List of passengers currently onboard
        zielstocke (set): Target floors to visit
        house (House): Reference to the House object
        idle_times (list): List of idle durations
        traveling_times (list): List of travel times
    """

    def __init__(self, env, aufzug_id, kapazitaet, house, fahrzeit=1.0, halte_zeit=0.5, check_interval_elevator=0.2):
        self.env = env
        self.id = aufzug_id
        self.kapazitaet = kapazitaet
        self.house = house
        self.akt_stock = 0
        self.richtung = 0
        self.fahrzeit = fahrzeit
        self.halte_zeit = halte_zeit
        self.check_interval = check_interval_elevator

        self.passagiere = []
        self.zielstocke = set()

        self.idle_since = env.now
        self.idle_times = []
        self.traveling_start = None
        self.traveling_times = []

    def __repr__(self):
        direction = "UP" if self.richtung == 1 else "DOWN" if self.richtung == -1 else "IDLE"
        return f"Elevator-{self.id}(floor={self.akt_stock}, {direction}, people={len(self.passagiere)})"

    def run(self):
        """
        Main simulation loop for the elevator.
        Continuously checks and processes passenger movements and destination floors.
        """
        while True:
            yield self.env.timeout(self.check_interval)

            # Try to process the current floor (pickup/dropoff)
            yield from self.process_current_floor()

            # If no targets and no pending requests -> idle
            if not self.zielstocke and not self.check_for_requests():
                if self.richtung != 0:
                    self.richtung = 0
                    self.idle_since = self.env.now
                continue

            # If elevator starts moving after idling, record idle time
            if self.richtung == 0:
                idle_duration = self.env.now - self.idle_since
                if idle_duration > 0:
                    self.idle_times.append(idle_duration)
                self.traveling_start = self.env.now

            yield from self.process_current_floor()
            self.determine_direction()

            if self.richtung != 0:
                yield from self.move_one_floor()

    def check_for_requests(self):
        """Returns True if there are any waiting people in the building."""
        return any(floor.elevatorQueueUp or floor.elevatorQueueDown for floor in self.house.floors)

    def process_current_floor(self):
        """
        Handles activities at the current floor:
          - Passengers exit
          - Passengers board
        """
        need_to_stop = False

        # Check if current floor is a destination
        if self.akt_stock in self.zielstocke:
            need_to_stop = True
            self.zielstocke.remove(self.akt_stock)

        # Check if passengers want to exit
        passengers_to_exit = [p for p in self.passagiere if p.targetFloor == self.akt_stock]
        if passengers_to_exit:
            need_to_stop = True

        # Check if people want to board
        current_floor = self.house.floors[self.akt_stock]
        if (self.richtung == 1 and current_floor.elevatorQueueUp) or \
           (self.richtung == -1 and current_floor.elevatorQueueDown) or \
           (self.richtung == 0 and (current_floor.elevatorQueueUp or current_floor.elevatorQueueDown)):
            need_to_stop = True

        if not need_to_stop:
            return

        # Simulate stop at the floor
        yield self.env.timeout(self.halte_zeit)

        # Passengers exit
        for passenger in passengers_to_exit[:]:
            self.passagiere.remove(passenger)
            passenger.currentFloor = self.akt_stock
            passenger.exit_elevator_time = self.env.now

            if passenger.enter_queue_time is not None and passenger.enter_elevator_time is not None:
                wait_time = passenger.enter_elevator_time - passenger.enter_queue_time
                self.house.waiting_times.append(wait_time)

            self.house.floors[self.akt_stock].idlePeople.append(passenger)
            logger.info(f"[t={self.env.now:.1f}] {passenger} exits {self} at floor {self.akt_stock}")

        # Passengers board
        if self.richtung >= 0 and current_floor.elevatorQueueUp:
            self.board_passengers(current_floor.elevatorQueueUp)
        if self.richtung <= 0 and current_floor.elevatorQueueDown:
            self.board_passengers(current_floor.elevatorQueueDown)

    def board_passengers(self, queue):
        """
        Boards passengers from the given queue, respecting elevator capacity.
        Updates target floors.
        """
        for person in queue.copy():
            if len(self.passagiere) >= self.kapazitaet:
                break

            queue.remove(person)
            person.currentFloor = -1
            person.enter_elevator_time = self.env.now

            self.passagiere.append(person)
            self.zielstocke.add(person.targetFloor)

            logger.info(f"[t={self.env.now:.1f}] {person} enters {self} at floor {self.akt_stock}, target={person.targetFloor}")

    def determine_direction(self):
        """
        Determines the next movement direction using the SCAN algorithm.
        Continues if possible; otherwise changes direction.
        """
        if not self.zielstocke and not self.check_for_requests():
            self.richtung = 0
            return

        if self.richtung == 1:
            floors_above = [f for f in self.zielstocke if f > self.akt_stock]
            if floors_above:
                self.richtung = 1
                return
            floors_below = [f for f in self.zielstocke if f < self.akt_stock]
            if floors_below:
                self.richtung = -1
                return

        elif self.richtung == -1:
            floors_below = [f for f in self.zielstocke if f < self.akt_stock]
            if floors_below:
                self.richtung = -1
                return
            floors_above = [f for f in self.zielstocke if f > self.akt_stock]
            if floors_above:
                self.richtung = 1
                return

        # If idle: pick nearest request
        if self.richtung == 0:
            closest_floor = None
            min_distance = float('inf')

            for i, floor in enumerate(self.house.floors):
                if floor.elevatorQueueUp or floor.elevatorQueueDown:
                    distance = abs(i - self.akt_stock)
                    if distance < min_distance:
                        min_distance = distance
                        closest_floor = i

            if closest_floor is not None:
                self.richtung = 1 if closest_floor > self.akt_stock else -1

        if self.richtung == 0 and self.zielstocke:
            if any(f > self.akt_stock for f in self.zielstocke):
                self.richtung = 1
            elif any(f < self.akt_stock for f in self.zielstocke):
                self.richtung = -1

    def move_one_floor(self):
        """
        Moves the elevator exactly one floor in its current direction.
        Updates position and checks for top/bottom limits.
        """
        if self.richtung == 0:
            return

        if self.traveling_start is not None:
            self.traveling_times.append(self.env.now - self.traveling_start)
            self.traveling_start = None

        # Simulate movement
        yield self.env.timeout(self.fahrzeit)

        self.akt_stock += self.richtung

        # Boundary checks
        if self.akt_stock < 0:
            self.akt_stock = 0
            self.richtung = 0
        elif self.akt_stock >= len(self.house.floors):
            self.akt_stock = len(self.house.floors) - 1
            self.richtung = 0

        logger.info(f"[t={self.env.now:.1f}] {self} moved to floor {self.akt_stock}")