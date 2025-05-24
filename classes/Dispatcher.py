import logging
import simpy

# Logger setup
logger = logging.getLogger(__name__)

class Dispatcher:
    """
    Dispatcher for basic elevator request handling.
      - Periodically checks all floors for pending requests.
      - Forwards each request to all elevators.
      - Elevators individually decide whether and when to handle forwarded requests (using SCAN).
    """

    def __init__(self, env, house, check_interval=0.2):
        """
        Initialize the dispatcher.
        env (simpy.Environment): The simulation environment.
        house (House): Reference to the building (floors + elevators).
        check_interval (float): Time interval between dispatch checks.
        """
        self.env = env
        self.house = house
        self.check_interval = check_interval

    def run(self):
        """
        Main dispatcher loop.
        Periodically triggers dispatching of floor requests to elevators.
        """
        while True:
            yield self.env.timeout(self.check_interval)
            self.dispatch_requests()

    def dispatch_requests(self):
        """
        Checks each floor for waiting passengers (up or down).
        If a request exists, forwards it to all elevators.
        """
        for floor_idx, floor in enumerate(self.house.floors):
            # Forward UP requests
            if floor.elevatorQueueUp.items:
                self.forward_request(floor_idx)

            # Forward DOWN requests
            if floor.elevatorQueueDown.items:
                self.forward_request(floor_idx)

    def forward_request(self, floor_idx):
        """
        Forwards a floor request to all elevators.
            floor_idx (int): The floor number where the request originated.
        """
        for elevator in self.house.elevators:
            # If elevator doesn't already know about this floor, add it
            if floor_idx not in elevator.zielstoecke:
                elevator.add_target(floor_idx)
                logger.info(f"[t={self.env.now:.1f}] Dispatcher forwarded request from floor {floor_idx} to {elevator}")
