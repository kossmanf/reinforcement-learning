import simpy

class Floor:
    """
    Represents a floor in the building.

    Attributes:
        floor_number (int): The floor's number (or index).
        elevatorQueueUp (list): List of people waiting to go UP.
        elevatorQueueDown (list): List of people waiting to go DOWN.
        idlePeople (list): List of people currently idle (no destination).
    """

    def __init__(self, env, floor_number: int):
        self.floor_number = floor_number
        self.elevatorQueueUp =  simpy.Store(env) # People waiting to go up
        self.elevatorQueueDown =  simpy.Store(env)  # People waiting to go down
        self.idlePeople = simpy.Store(env) # People not currently requesting transport

    def __repr__(self):
        """
        String representation of the floor.
        Example: 'Floor-3'
        """
        return f"Floor-{self.floor_number}"
