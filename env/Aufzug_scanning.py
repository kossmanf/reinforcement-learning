import logging
import simpy

# Logger setup
logger = logging.getLogger(__name__)


class Aufzug_scanning:
    """
    Represents an elevator using the SCAN algorithm ("elevator scan").
    
    The elevator moves in one direction (up or down) until there are no more requests
    in that direction, then reverses direction.
    """

    def __init__(self, env, aufzug_id, kapazitaet, house, 
                 fahrzeit=1.0, halte_zeit=0.5, check_interval_elevator=0.2):
        """
        Initialize elevator with its properties and operational parameters.
        
        Args:
            env: SimPy environment
            aufzug_id: Unique identifier for this elevator
            kapazitaet: Maximum number of passengers
            house: Reference to the house containing the elevator
            fahrzeit: Time to travel between floors (seconds)
            halte_zeit: Time spent at a floor when stopping (seconds)
            check_interval_elevator: How often to check for new states (seconds)
        """
        # Environment and identification
        self.env = env
        self.id = aufzug_id
        self.kapazitaet = kapazitaet
        self.house = house
        
        # Movement parameters
        self.current_floor = 0
        self.direction = 0  # 0=idle, 1=up, -1=down
        self.next_target = None
        self.fahrzeit = fahrzeit
        self.halte_zeit = halte_zeit
        self.check_interval_elevator = check_interval_elevator

        # Passenger management
        self.passengers = []
        self.zielstoecke = set()

        # Statistics tracking
        self.idle_since = env.now
        self.idle_times = []
        self.traveling_start = None
        self.traveling_times = []

        logger.info(f"[t={self.env.now:.1f}] Elevator-{self.id} initialized at floor {self.current_floor}, kapazitaet={self.kapazitaet}")

    def __repr__(self):
        """String representation of the elevator's current state."""
        direction = "UP" if self.direction == 1 else "DOWN" if self.direction == -1 else "IDLE"
        return f"Elevator-{self.id}(floor={self.current_floor}, {direction}, people={len(self.passengers)})"

    def run(self):
        """Main operational process of the elevator."""
        while True:
            yield self.env.timeout(self.check_interval_elevator)
            yield self.env.process(self.wait_if_idle())
            yield self.env.process(self.process_current_floor())
            self.determine_next_target()
            self.determine_direction()
            yield self.env.process(self.move_one_floor())
            self.handle_boundary_direction_change()

    def wait_if_idle(self):
        """Wait in idle state until there are target floors to service."""
        if not self.zielstoecke:
            if self.direction != 0:
                logger.info(f"[t={self.env.now:.1f}] {self} has no targets, switching to IDLE")
            self.direction = 0
            self.idle_since = self.env.now
            yield from self.wait_until(lambda: len(self.zielstoecke) > 0)
            idle_duration = self.env.now - self.idle_since
            self.idle_times.append(idle_duration)
            logger.info(f"[t={self.env.now:.1f}] {self} resumes from IDLE after {idle_duration:.1f}s")

    def wait_until(self, condition, check_interval_elevator=0.2):
        """Helper method to wait until a condition is met."""
        while not condition():
            yield self.env.timeout(check_interval_elevator)

    def process_current_floor(self):
        """Handle operations when arriving at a floor (unload/load passengers)."""
        # Check if we need to stop at this floor
        need_to_stop = False
        if self.current_floor in self.zielstoecke:
            need_to_stop = True
            self.zielstoecke.discard(self.current_floor)
            logger.info(f"[t={self.env.now:.1f}] {self} reached target floor {self.current_floor}")
            self.next_target = None

        if not need_to_stop:
            return

        logger.info(f"[t={self.env.now:.1f}] {self} stopping at floor {self.current_floor}")
        
        # First unload passengers whose destination is this floor
        yield self.env.process(self.unload_passengers())
        
        # Then load waiting passengers
        up_queue = self.house.floors[self.current_floor].elevatorQueueUp
        yield self.env.process(self.load_passengers(up_queue))
        
        down_queue = self.house.floors[self.current_floor].elevatorQueueDown
        yield self.env.process(self.load_passengers(down_queue))
        
        # Wait for stop time
        yield self.env.timeout(self.halte_zeit)

    def unload_passengers(self):
        """Remove passengers who have reached their destination."""
        exiting = [p for p in self.passengers if p.targetFloor == self.current_floor]
        for passenger in exiting:
            self.passengers.remove(passenger)
            passenger.currentFloor = self.current_floor
            passenger.exit_elevator_time = self.env.now
            yield self.house.floors[self.current_floor].idlePeople.put(passenger)
            
            ride_time = passenger.exit_elevator_time - passenger.enter_elevator_time
            self.traveling_times.append(ride_time)
            logger.info(f"[t={self.env.now:.1f}] {passenger} exited {self} at floor {self.current_floor} "
                        f"(ride time: {ride_time:.1f}s)")

    def load_passengers(self, queue):
        """Load waiting passengers from a queue until kapazitaet is reached."""
        while len(self.passengers) < self.kapazitaet and queue.items:
            person = yield queue.get()
            self.passengers.append(person)
            self.add_target(person.targetFloor)

            person.enter_elevator_time = self.env.now
            wait_time = person.enter_elevator_time - person.enter_queue_time
            self.house.waiting_times.append(wait_time)

            logger.info(f"[t={self.env.now:.1f}] {person} entered {self} at floor {self.current_floor}, "
                        f"target={person.targetFloor} (waited {wait_time:.1f}s)")

    def determine_next_target(self):
        """Determine the next floor to visit based on the SCAN algorithm."""
        previous = self.next_target
        up_targets = [t for t in self.zielstoecke if t > self.current_floor]
        down_targets = [t for t in self.zielstoecke if t < self.current_floor]

        if self.direction == 1:  # Moving up
            self.next_target = min(up_targets) if up_targets else (max(down_targets) if down_targets else None)
        elif self.direction == -1:  # Moving down
            self.next_target = max(down_targets) if down_targets else (min(up_targets) if up_targets else None)
        else:  # Idle
            # When idle, choose the closest target
            if up_targets and down_targets:
                up_distance = min(up_targets) - self.current_floor
                down_distance = self.current_floor - max(down_targets)
                self.next_target = min(up_targets) if up_distance <= down_distance else max(down_targets)
            else:
                self.next_target = min(up_targets) if up_targets else (max(down_targets) if down_targets else None)

        if self.next_target != previous:
            logger.info(f"[t={self.env.now:.1f}] {self} next target updated to {self.next_target}")
    
    def determine_direction(self):
        """
        Set movement direction based on the SCAN algorithm principles.
        Direction only changes when there are no more targets in the current direction
        or the elevator is idle.
        """
        if self.next_target is None:
            logger.info(f"[t={self.env.now:.1f}] {self} has no next target, setting direction to IDLE")
            self.direction = 0
            return
            
        # If currently idle, set direction based on next target
        if self.direction == 0:
            if self.next_target > self.current_floor:
                self.direction = 1  # UP
                logger.info(f"[t={self.env.now:.1f}] {self} was IDLE, setting direction to UP "
                          f"(current floor: {self.current_floor}, target: {self.next_target})")
            elif self.next_target < self.current_floor:
                self.direction = -1  # DOWN
                logger.info(f"[t={self.env.now:.1f}] {self} was IDLE, setting direction to DOWN "
                          f"(current floor: {self.current_floor}, target: {self.next_target})")
            return
            
        # Check if we need to change direction based on available targets
        up_targets = [t for t in self.zielstoecke if t > self.current_floor]
        down_targets = [t for t in self.zielstoecke if t < self.current_floor]
        
        # If moving up and no more targets above, change direction
        if self.direction == 1 and not up_targets:
            if down_targets:  # Only change if there are targets below
                self.direction = -1
                logger.info(f"[t={self.env.now:.1f}] {self} no more UP targets, changing direction to DOWN")
        
        # If moving down and no more targets below, change direction
        elif self.direction == -1 and not down_targets:
            if up_targets:  # Only change if there are targets above
                self.direction = 1
                logger.info(f"[t={self.env.now:.1f}] {self} no more DOWN targets, changing direction to UP")

    def add_target(self, floor_number):
        """Add a floor to the target list if it's valid."""
        if 0 <= floor_number < len(self.house.floors):
            if floor_number not in self.zielstoecke:
                logger.info(f"[t={self.env.now:.1f}] {self} added new target floor {floor_number}")
                self.zielstoecke.add(floor_number)

    def handle_boundary_direction_change(self):
        """Reverse direction if at house boundaries."""
        # Simply enforce the boundary conditions
        if self.direction == 1 and self.current_floor == len(self.house.floors) - 1:
            # At top floor, must go down or idle
            if any(f < self.current_floor for f in self.zielstoecke):
                self.direction = -1
                logger.info(f"[t={self.env.now:.1f}] {self} hit top floor, reversing direction to DOWN")
            else:
                self.direction = 0
                logger.info(f"[t={self.env.now:.1f}] {self} hit top floor with no more targets, going IDLE")
                
        elif self.direction == -1 and self.current_floor == 0:
            # At bottom floor, must go up or idle
            if any(f > self.current_floor for f in self.zielstoecke):
                self.direction = 1
                logger.info(f"[t={self.env.now:.1f}] {self} hit bottom floor, reversing direction to UP")
            else:
                self.direction = 0
                logger.info(f"[t={self.env.now:.1f}] {self} hit bottom floor with no more targets, going IDLE")

    def move_one_floor(self):
        """Move the elevator one floor in the current direction."""
        if self.next_target is None:
            logger.info(f"[t={self.env.now:.1f}] {self} has no next target, staying on floor {self.current_floor}")
            return
        
        if self.current_floor == self.next_target:
            logger.info(f"[t={self.env.now:.1f}] {self} is already on target floor {self.next_target}, not moving")
            return

        if self.direction == 1:  # Moving up
            logger.info(f"[t={self.env.now:.1f}] {self} moving UP from floor {self.current_floor} "
                      f"to floor {self.current_floor + 1} (target: {self.next_target})")
            self.traveling_start = self.env.now
            yield self.env.timeout(self.fahrzeit)
            self.current_floor += 1

        elif self.direction == -1:  # Moving down
            logger.info(f"[t={self.env.now:.1f}] {self} moving DOWN from floor {self.current_floor} "
                      f"to floor {self.current_floor - 1} (target: {self.next_target})")
            self.traveling_start = self.env.now
            yield self.env.timeout(self.fahrzeit)
            self.current_floor -= 1

        elif self.direction == 0:  # Idle
            logger.info(f"[t={self.env.now:.1f}] {self} is IDLE, staying on floor {self.current_floor}")