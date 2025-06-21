import numpy as np

def get_person_status(person, house):
    """
    Determines a person's current status in the building:
      - "IN_ELEVATOR" if the person is inside any elevator,
      - "WAITING (UP)" if waiting to go up,
      - "WAITING (DOWN)" if waiting to go down,
      - "IDLE" if idle at some floor,
      - "UNKNOWN" otherwise.
    """
    for elev in house.elevators:
        if person in elev.passengers:
            return "IN_ELEVATOR"
    for floor in house.floors:
        if person in floor.elevatorQueueUp.items:
            return "WAITING (UP)"
        if person in floor.elevatorQueueDown.items:
            return "WAITING (DOWN)"
        if person in floor.idlePeople.items:
            return "IDLE"
    return "UNKNOWN"

def get_elevator_status(elev):
    """
    Returns the elevator's direction/state:
      - "UP" if moving up,
      - "DOWN" if moving down,
      - "IDLE" if not moving.
    """
    if elev.direction == 1:
        return "UP"
    elif elev.direction == -1:
        return "DOWN"
    else:
        return "IDLE"

def draw_simulation_step(fig, house, current_time, bg_image=None, fontsize=5):
    """
    Visualizes a snapshot of the simulation at a given time.

    Shows:
      - People on floors
      - Elevator positions and directions
      - Current floor destination probabilities
      - Sample of exponential interarrival times
    """
    # Axes setup
    ax_left = fig.axes[0]
    ax_right = fig.axes[1]
    ax_floor_dist = fig.axes[2]
    ax_spawn_dist = fig.axes[3]

    elev_ids = [elev.id for elev in house.elevators]
    elev_floors = [elev.current_floor for elev in house.elevators]

    XMIN, YMIN = 0, 0
    YMAX = house.num_floors + 1

    # --- Prepare people plot data ---
    floor_to_people = dict()
    for person in house.personList:
        floor = person.currentFloor
        floor_to_people.setdefault(floor, []).append(person)

    person_plot_x = []
    person_plot_y = []
    person_labels = []

    for floor, persons in floor_to_people.items():
        for idx, person in enumerate(persons):
            person_plot_x.append(idx)                      # Person index on this floor
            person_plot_y.append(floor)                    # Floor number
            person_labels.append(f"{person.person_id}")    # Person ID

    # --- Plot people ---
    ax_left.cla()
    ax_left.set_title(f"People (t = {current_time})")
    ax_left.set_xlabel("Persons on Floor")
    ax_left.set_ylabel("Floor")

    # Set dynamic limits
    MIN_XMAX = 10
    dynamic_xmax = max(person_plot_x) + 2 if person_plot_x else 0
    final_xmax = max(dynamic_xmax, MIN_XMAX)

    ax_left.set_xlim(XMIN, final_xmax)
    ax_left.set_ylim(YMIN, YMAX)

    if bg_image is not None:
        ax_left.imshow(bg_image, extent=[XMIN, final_xmax, YMIN, YMAX], origin="upper", alpha=0.3, zorder=0)

    # Draw floor lines
    for fl in range(YMIN, YMAX + 1):
        ax_left.axhline(y=fl, color="white", alpha=1, zorder=1)

    ax_left.scatter(person_plot_x, person_plot_y, marker="o", color="red", zorder=2)

    # Annotate each person with ID and status
    for i in range(len(person_plot_x)):
        floor = person_plot_y[i]
        person_id = int(person_labels[i])
        person = next(p for p in house.personList if p.person_id == person_id)
        status = get_person_status(person, house)
        ax_left.text(person_plot_x[i] + 0.1, person_plot_y[i] + 0.2, f"{person_labels[i]} ({status})", fontsize=fontsize, color="black", zorder=3)

    # --- Plot elevators ---
    ax_right.cla()
    ax_right.set_title(f"Elevators (t = {current_time})")
    ax_right.set_xlabel("Elevator-ID")
    ax_right.set_ylabel("Floor")

    MIN_XMAX_ELEV = 10
    dynamic_xmax_elev = max(elev_ids) + 2 if elev_ids else 0
    final_xmax_elev = max(dynamic_xmax_elev, MIN_XMAX_ELEV)

    ax_right.set_xlim(0, final_xmax_elev)
    ax_right.set_ylim(YMIN, YMAX)

    if bg_image is not None:
        ax_right.imshow(bg_image, extent=[0, final_xmax_elev, YMIN, YMAX], origin="upper", alpha=0.3, zorder=0)

    for fl in range(YMIN, YMAX + 1):
        ax_right.axhline(y=fl, color="white", alpha=1, zorder=1)

    ax_right.scatter(elev_ids, elev_floors, marker="s", color="yellow", zorder=2)

    # Annotate elevators
    for i, eid in enumerate(elev_ids):
        elev = house.elevators[i]
        status = get_elevator_status(elev)
        ax_right.text(eid + 0.3, elev_floors[i] + 0.3, f"E{eid} ({status})", fontsize=fontsize, color="black", zorder=3)

    # --- Plot floor target distribution ---
    ax_floor_dist.cla()
    ax_floor_dist.set_title(f"Floor Target Probabilities (t = {current_time})")
    ax_floor_dist.set_xlabel("Target Floor")
    ax_floor_dist.set_ylabel("Probability")

    time_index = int(current_time)
    probs = house.floor_distribution[time_index] if time_index < len(house.floor_distribution) else house.floor_distribution[-1]

    ax_floor_dist.bar(range(len(probs)), probs, color='skyblue')

    # --- Plot sampled exponential spawn times ---
    ax_spawn_dist.cla()
    ax_spawn_dist.set_title(f"Sampled Spawn Times (Exponential) (t = {current_time})")
    ax_spawn_dist.set_xlabel("Interarrival Time")
    ax_spawn_dist.set_ylabel("Frequency")

    avg_arrival = house.averageArrivalTimes[time_index] if time_index < len(house.averageArrivalTimes) else house.averageArrivalTimes[-1]

    if avg_arrival > 0:
        samples = np.random.exponential(avg_arrival, size=1000)
        ax_spawn_dist.hist(samples, bins=30, color='green', edgecolor='black')
    else:
        ax_spawn_dist.text(0.5, 0.5, "λ = 0 (No spawn)", ha='center', va='center', transform=ax_spawn_dist.transAxes)

    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_metrics_over_time(metrics, fig):
    """
    Plots key simulation metrics over time:
      - Average waiting time
      - Average traveling time
      - Average elevator idle time
      - Average time inside the building
    """
    if not metrics or not metrics["time"]:
        print("No metric data available.")
        return

    ax = fig.add_subplot(1, 1, 1)

    # Plot each metric
    ax.plot(metrics["time"], metrics["avg_waiting"], label="Avg Waiting Time")
    ax.plot(metrics["time"], metrics["avg_traveling"], label="Avg Traveling Time")
    ax.plot(metrics["time"], metrics["avg_idle"], label="Avg Idle Time")
    ax.plot(metrics["time"], metrics["avg_building"], label="Avg Time in Building")

    # Set labels and formatting
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Time Units")
    ax.set_title("Average Metrics Over Time")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.show()
