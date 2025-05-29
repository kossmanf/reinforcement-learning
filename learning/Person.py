import logging

logger = logging.getLogger(__name__)

class Person:
    """
    Represents a person in the building.

    Attributes:
        person_id (int): Unique ID of the person
        startFloor (int): Floor where the person spawned
        targetFloor (int): Floor the person wants to reach
        spawn_time (float): SimPy time when the person was created
        currentFloor (int): Current floor (if not inside the elevator)
    """

    def __init__(self, person_id: int, startFloor: int, targetFloor: int, spawn_time: float):
        # Initialize a new person with their start and target floors
        self.person_id = person_id
        self.startFloor = startFloor
        self.targetFloor = targetFloor
        self.spawn_time = spawn_time

        self.currentFloor = startFloor

        # Timestamps for performance metrics
        self.enter_queue_time = None       # Time when the person entered the elevator queue
        self.enter_elevator_time = None    # Time when the person entered the elevator
        self.exit_elevator_time = None     # Time when the person exited the elevator
        self.leave_building_time = None    # Time when the person left the building

    def __repr__(self):
        # Return a string representation for logging and debugging
        return f"Person-{self.person_id}(start={self.startFloor}, target={self.targetFloor}, now={self.currentFloor})"
