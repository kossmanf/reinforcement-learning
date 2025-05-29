import logging

logger = logging.getLogger(__name__)

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
        self.direction = 0  # 0=idle, 1=up, -1=down
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

    def __repr__(self):
        direction = "UP" if self.direction == 1 else "DOWN" if self.direction == -1 else "IDLE"
        return f"Elevator-{self.id}(floor={self.current_floor}, {direction}, people={len(self.passengers)})"

    '''
    def run(self, decision_callback):
        while True:
            state = self.get_state()
            action = decision_callback(state)
            yield self.env.process(self.execute_action(action))
    '''

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
        unloading_reward = 0
        loading_reward = 0
        
        '''
        if action == "idle":
            if self.direction != 0:
                self.idle_since = self.env.now
            self.direction = 0
            yield self.env.timeout(self.check_interval_elevator )
        '''

        if action == "up":
            if self.direction == 0:
                time_now = self.env.now
                self.idle_times.append(time_now - self.idle_since)

            if self.current_floor < len(self.house.floors) - 1:
                # not at top floor
                self.direction = 1
                self.current_floor += 1
                #yield self.env.timeout(self.fahrzeit)

        elif action == "down":
            if self.direction == 0:
                    time_now = self.env.now
                    self.idle_times.append(time_now - self.idle_since)
        
            if self.current_floor > 0:
                # not at bottom floor
                self.direction = -1
                self.current_floor -= 1
                #yield self.env.timeout(self.fahrzeit)
                
        elif action == "stop":
            self.zielstoecke.discard(self.current_floor)
            unloading_reward  = yield self.env.process(self.unload_passengers())
            loading_reward = yield self.env.process(self.load_passengers())
            #yield self.env.timeout(self.halte_zeit)

            if not self.is_waiting() and self.is_empty():
                '''going into idle'''
                if self.direction != 0:
                    self.idle_since = self.env.now

                self.direction = 0
    
        reward = unloading_reward + loading_reward
        self.last_reward = reward   

    def unload_passengers(self):
        """
        Unload passengers whose destination is the current floor.
        """
        exiting = [p for p in self.passengers if p.targetFloor == self.current_floor]

        reward = 0

        for passenger in exiting:
            self.passengers.remove(passenger)
            passenger.currentFloor = self.current_floor
            passenger.exit_elevator_time = self.env.now
            yield self.house.floors[self.current_floor].idlePeople.put(passenger)
            ride_time = passenger.exit_elevator_time - passenger.enter_elevator_time
            self.traveling_times.append(ride_time)

            reward += self.get_reward(300, ride_time)
        
        return reward

    def load_passengers(self):
        """
        Load passengers based on elevator direction.
        - UP: only from elevatorQueueUp
        - DOWN: only from elevatorQueueDown
        - IDLE: from either queue
        """
        floor = self.house.floors[self.current_floor]
        reward = 0
        
        queue = None

        assert 0 <= self.current_floor < len(self.house.floors), f"Invalid floor: {self.current_floor}"


        # Choose which queues to serve
        if self.direction == 1:
            queue = floor.elevatorQueueUp
        elif self.direction == -1:
            queue = floor.elevatorQueueDown
        elif self.direction == 0:
            if len(floor.elevatorQueueUp.items) > 0:
                queue = floor.elevatorQueueUp
            elif len(floor.elevatorQueueDown.items) > 0:
                queue = floor.elevatorQueueDown

        if queue is None:
            return 0
        
        while len(self.passengers) < self.kapazitaet and len(queue.items) > 0:
            person = yield queue.get()
            self.passengers.append(person)
            self.zielstoecke.add(person.targetFloor)
            person.enter_elevator_time = self.env.now
            wait_time = person.enter_elevator_time - person.enter_queue_time
            self.house.waiting_times.append(wait_time)
            self.waiting_times.append(wait_time)

            reward += self.get_reward(300, wait_time)
        
        return reward
        
    def get_reward(self, max_time, time):
        return max(max_time - time, 1)

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