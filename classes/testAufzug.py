import simpy
import logging
from Aufzug import Aufzug
from Dispatcher import Dispatcher

import simpy
import logging

# Logging aktivieren
logging.basicConfig(level=logging.INFO)

# Dummy-Person
class Person:
    def __init__(self, name, start_floor, target_floor):
        self.name = name
        self.currentFloor = start_floor
        self.targetFloor = target_floor
        self.enter_queue_time = None
        self.enter_elevator_time = None
        self.exit_elevator_time = None

    def __repr__(self):
        return f"{self.name}(→{self.targetFloor})"

# Dummy-Floor
class Floor:
    def __init__(self, env):
        self.elevatorQueueUp = simpy.Store(env)
        self.elevatorQueueDown = simpy.Store(env)
        self.idlePeople = simpy.Store(env)

# Dummy-House mit 5 Stockwerken
class House:
    def __init__(self, num_floors, env):
        self.floors = [Floor(env) for _ in range(num_floors)]
        self.elevators = []
        self.waiting_times = []

# Dein ursprünglicher Aufzug (ausgelassen, da du ihn schon gegeben hast)

# Testumgebung mit Dispatcher und Aufzug
def test_elevator_with_dispatcher():
    env = simpy.Environment()
    house = House(5,env)

    # Personen erzeugen
    p1 = Person("Alice", 0, 3)
    p2 = Person("Bob", 2, 0)
    
    house.floors[0].elevatorQueueUp.put(p1)
    house.floors[2].elevatorQueueDown.put(p2)

    p1.enter_queue_time = 0
    p2.enter_queue_time = 0

    aufzug = Aufzug(env, aufzug_id=1, kapazitaet=4, house=house)
    house.elevators.append(aufzug)

    dispatcher = Dispatcher(env, house)

    # Prozesse starten
    env.process(aufzug.run())
    env.process(dispatcher.run())

    # Simulation
    env.run(until=10)

    # Ergebnisse
    print("Wartezeiten:", house.waiting_times)
    print("Reisezeiten:", aufzug.traveling_times)
    print("Leerlaufzeiten:", aufzug.idle_times)

if __name__ == "__main__":
    test_elevator_with_dispatcher()
