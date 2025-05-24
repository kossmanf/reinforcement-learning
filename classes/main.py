import tkinter as tk
import logging
from SimulationGUI import SimulationGUI

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("simulation.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(ch)

    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
