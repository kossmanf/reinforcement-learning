import numpy as np

class ElevatorController:
    def __init__(self, env, model, gym_env):
        self.env = env
        self.model = model
        self.gym_env = gym_env

    
    def elevator_controller(self, interval):
        """
        SimPy process that runs every `interval` to control elevators based on the trained model.
        """
        while True:
            # Get current state
            house_state = self.gym_env.get_house_state()
            elev_states = self.gym_env.get_elev_states()
            obs = self.gym_env.get_observation(house_state, elev_states)

            # Split observations per elevator
            splitted_obs = self.split_observation_by_elevator(
                obs, self.gym_env.num_elevators, self.gym_env.num_floors
            )

            # Get the full action mask for all elevators (shape: num_elevators x 3)
            full_mask = self.gym_env.getMask()

            # Loop through elevators and make individual decisions
            for i, obs_per_elevator in enumerate(splitted_obs):
                action_mask = full_mask[i]  

                # Predict action using model and action mask
                action, _ = self.model.predict(
                    obs_per_elevator,
                    deterministic=True,
                    action_masks=action_mask
                )

                # Decode action ID to string
                action_str = {0: "up", 1: "down", 2: "stop"}[int(action)]

                # Dispatch action in the SimPy environment
                self.env.process(self.gym_env.house.elevators[i].execute_action(action_str))

            # Wait until the next control step
            yield self.env.timeout(interval)

    def split_observation_by_elevator(self, obs, num_elevators, num_floors):
        idx = 0
        result = []
        cap = self.gym_env.elevator_capacity

        elevator_blocks = []
        for _ in range(num_elevators):
            floor_vec = list(obs[idx:idx + num_floors])
            idx += num_floors

            dir_vec = list(obs[idx:idx + 3])
            idx += 3

            pass_vec = list(obs[idx:idx + cap])
            idx += cap

            target_vec = list(obs[idx:idx + num_floors])
            idx += num_floors

            elevator_obs = floor_vec + dir_vec + pass_vec + target_vec
            elevator_blocks.append(elevator_obs)

        # Shared floor info (once for all elevators)
        waiting_up = []
        waiting_down = []
        for _ in range(num_floors):
            waiting_up.append(obs[idx])
            waiting_down.append(obs[idx + 1])
            idx += 2

        # Combine each elevator’s obs with shared floor info
        for elev_obs in elevator_blocks:
            result.append(np.array(elev_obs + waiting_up + waiting_down, dtype=np.int32))

        return result
