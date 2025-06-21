import logging
import os

os.makedirs("../logs", exist_ok=True)

elevator_logger = logging.getLogger("elevator")
elevator_logger.setLevel(logging.INFO)

if not elevator_logger.handlers:
    file_handler = logging.FileHandler("../logs/elevator.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    elevator_logger.addHandler(file_handler)

class Aufzug:
    """
    Elevator controlled by a reinforcement learning agent.
    The agent decides direction, stopping, and idle behavior.
    """

    def __init__(self, env, aufzug_id, kapazitaet, house, 
                 fahrzeit=1.0, halte_zeit=0.5, check_interval_elevator=0.2):
        self.env = env
        self.id = aufzug_id
        self.kapazitaet = kapazitaet
        self.house = house
        self.current_floor = 0
        self.direction = 0  
        self.fahrzeit = fahrzeit
        self.halte_zeit = halte_zeit
        self.check_interval_elevator = check_interval_elevator

        self.passengers = []
        self.zielstoecke = set()

        self.idle_times = []
        self.traveling_times = []
        self.waiting_times = []

        self.idle_since = env.now

        self.last_reward = 0

        self.weight_unloading = 0.8
        self.weight_loading = 0.2
        self.weight_idle = 0

        self.unloading_reward_value = 15
        self.loading_reward_value = 5
        self.idle_reward_value = 0

    def __repr__(self):
        direction = "UP" if self.direction == 1 else "DOWN" if self.direction == -1 else "IDLE"
        return f"Elevator-{self.id}(floor={self.current_floor}, {direction}, people={len(self.passengers)})"

    def get_state(self):
        """
        Returns current state as dict for RL agent.
        """
        return {
            "floor": self.current_floor,
            "direction": self.direction,
            "passenger_count": len(self.passengers),
            "zielstoecke": list(self.zielstoecke),
            "waiting_up": [len(f.elevatorQueueUp.items) for f in self.house.floors],
            "waiting_down": [len(f.elevatorQueueDown.items) for f in self.house.floors]
        }

    def execute_action(self, action):
        """
        Executes an action string provided by the agent.
        Valid actions: "up", "down", "stop"
        """
        yield self.env.timeout(self.check_interval_elevator)

        unloading_reward = 0
        loading_reward = 0
        idle_reward = 0
        
        if action == "up":
            if self.direction == 0:
                time_now = self.env.now
                self.idle_times.append(time_now - self.idle_since)

            if self.current_floor < len(self.house.floors) - 1:
                # not at top floor
                self.direction = 1
                self.current_floor += 1
                log_msg = "went up"
                elevator_logger.info(log_msg)
                yield self.env.timeout(self.fahrzeit)

            # direction change
            if self.current_floor == len(self.house.floors) -1:
                self.direction = - 1

        elif action == "down":
            if self.direction == 0:
                    time_now = self.env.now
                    self.idle_times.append(time_now - self.idle_since)
        
            if self.current_floor > 0:
                # not at bottom floor
                self.direction = -1
                self.current_floor -= 1
                log_msg = "went down"
                elevator_logger.info(log_msg)
                yield self.env.timeout(self.fahrzeit)
            
            # direction change
            if self.current_floor == 0:
                self.direction = 1
            
                
        elif action == "stop":
            self.zielstoecke.discard(self.current_floor)
            unloading_reward  = yield self.env.process(self.unload_passengers())
            loading_reward = yield self.env.process(self.load_passengers())

            if not self.is_waiting() and not self.has_pending_targets() and self.is_empty():
                '''going into idle'''
                if self.direction != 0:
                    self.idle_since = self.env.now
                    idle_reward = self.idle_reward_value
                self.direction = 0
                log_msg = "went idle"
                elevator_logger.info(log_msg)
            else:
                yield self.env.timeout(self.halte_zeit)
    
        reward = (self.weight_unloading * unloading_reward +
                self.weight_loading * loading_reward +
                self.weight_idle * idle_reward)
        self.last_reward = reward   

        log_msg = (
            f"[t={self.env.now:.1f}] Elevator-{self.id} | "
            f"Action: {action} | Floor: {self.current_floor} | "
            f"Reward: {reward:.2f} (Unload: {unloading_reward}, Load: {loading_reward}, Idle: {idle_reward}) | "
            f"Direction: {self.direction} | Passengers: {len(self.passengers)} | "
            f"waiting_up: {','.join(str(len(f.elevatorQueueUp.items)) for f in self.house.floors)} | "
            f"waiting_down: {','.join(str(len(f.elevatorQueueDown.items)) for f in self.house.floors)}"
        )
        elevator_logger.info(log_msg)

    def unload_passengers(self):
        """
        Unload passengers whose destination is the current floor.
        """
        log_msg = "unloading passangers"
        elevator_logger.info(log_msg)
        
        exiting = [p for p in self.passengers if p.targetFloor == self.current_floor]

        reward = 0

        unloaded = False
        for passenger in exiting:
            self.passengers.remove(passenger)
            passenger.currentFloor = self.current_floor
            passenger.exit_elevator_time = self.env.now
            yield self.house.floors[self.current_floor].idlePeople.put(passenger)
            ride_time = passenger.exit_elevator_time - passenger.enter_elevator_time
            self.traveling_times.append(ride_time)
            unloaded = True

        log_msg = "unloaded passangers"
        elevator_logger.info(log_msg)

        if unloaded:
            reward = self.unloading_reward_value
        
        return reward

    def load_passengers(self):
        """
        Load passengers based on elevator direction.
        - UP: only from elevatorQueueUp
        - DOWN: only from elevatorQueueDown
        - IDLE: from either queue
        """
    
        log_msg = "loading passangers"
        elevator_logger.info(log_msg)
        floor = self.house.floors[self.current_floor]
        reward = 0
        
        assert 0 <= self.current_floor < len(self.house.floors), f"Invalid floor: {self.current_floor}"

        queue = floor.elevatorQueueUp

        if self.direction == -1:
            queue = floor.elevatorQueueDown
        
        loaded = False
        while len(self.passengers) < self.kapazitaet and len(queue.items) > 0:
            person = yield queue.get()
            self.passengers.append(person)
            self.zielstoecke.add(person.targetFloor)
            person.enter_elevator_time = self.env.now
            wait_time = person.enter_elevator_time - person.enter_queue_time
            self.house.waiting_times.append(wait_time)
            self.waiting_times.append(wait_time)
            loaded = True
        
        log_msg = "loaded passangers"
        elevator_logger.info(log_msg)

        if loaded:
            reward = self.loading_reward_value

        return reward

    def is_empty(self):
        return len(self.passengers) == 0

    def is_waiting(self):
        """
        Check if any person is waiting in any elevator queue in the building.
        Returns True if at least one person is waiting.
        """
        return any(
            len(f.elevatorQueueUp.items) > 0 or len(f.elevatorQueueDown.items) > 0
            for f in self.house.floors
        )
    
    def has_pending_targets(self):
        return len(self.zielstoecke) > 0