import numpy as np
from matplotlib import pyplot as plt

from control_ablations.generic_targets import NoLearningControlTarget
from control_ablations.generic_targets.oc_io import OCIO
from control_ablations.generic_targets.optimal_control import OptimalControlCollocation
import time


class OptimalControlTarget(NoLearningControlTarget):

    def __init__(self, trial_settings):
        super().__init__(trial_settings)
        # No actual iterations but keeps everything aligned...
        self.io = OCIO(self.test_name)
        self.time = 0
        self.action_sequence = None
        self.env_actions = None
        self.optimal_control = None
        self.casadi_settings = {}
        self.num_states = 0
        self.num_controls = 0

    # The "training" phase in optimal control is where it calculates the static
    # expected optimum control for the equivalent continuous system.
    # Ignores anything around iterations because it doesn't make sense to do the
    # optimisation multiple times. Could run from different starting points but
    # it's sensitive to initialisation so starting from a random number is not a
    # great plan.
    def train(self):
        self.io.make_training_file_structure()
        self.get_casadi_settings()

        # Solve optimal control.
        start_time = time.process_time()
        # use_midpoint = self.controller_settings.get("use_midpoint", True)
        optimiser = OptimalControlCollocation(**self.casadi_settings,
                                              initialisation_plot_path=self.io.optimiser_init_plot_path())

        # Check output and stop if optimiser has not converged.
        success, self.optimal_control, state = optimiser.run()

        if not success:
            print("Optimiser did not converge. Halting training")
            assert 0
        print("Completed optimisation.")

        optimiser.plot(state_line_type=['.']*self.num_states,
                       control_line_type=['-.']*self.num_controls,
                       filepath=self.io.optimiser_plot_path())

        self.env_actions = self.convert_to_env_actions(self.optimal_control, state)
        end_time = time.process_time()

        # Record various outputs for debug and plotting
        self.plot_env_actions()
        # Save the control trajectories to a file.
        self.io.save_trajectory(self.env_actions)
        state_trajectory = self.convert_to_env_states(state)
        self.io.save_state_trajectory(state_trajectory)

        self.io.dump_training_time(str(end_time - start_time))

    def get_casadi_settings(self):
        # Get system dynamics and initial state.
        if "casadi_settings" in self.controller_settings:
            print("Using supplied dynamics for optimisation")
            self.casadi_settings = self.controller_settings["casadi_settings"]
        else:
            # This should be the main use case.
            self.casadi_settings = self.generate_dynamics_from_sim_setup()
        self.num_states = len(self.casadi_settings['state_list'])
        self.num_controls = len(self.casadi_settings['control_list'])
        self.check_casadi_settings()

    def check_casadi_settings(self):
        # Basic checks on bounds:
        assert (self.num_states == len(self.casadi_settings['dynamics_list']))
        assert (self.num_states == len(self.casadi_settings['initial_state']))
        assert (self.num_states == len(self.casadi_settings['state_bounds_lower']))
        assert (self.num_states == len(self.casadi_settings['state_bounds_upper']))
        assert (self.num_controls == len(self.casadi_settings['control_bounds_upper']))
        assert (self.num_controls == len(self.casadi_settings['control_bounds_lower']))
        assert (self.num_controls == len(self.casadi_settings['control_initial_trajectories'][0]))

    def plot_env_actions(self):
        fig, ax = plt.subplots(1)
        ax.plot(self.env_actions.transpose())
        fig.savefig(self.io.env_plot_path(self.time))

    def generate_dynamics_from_sim_setup(self):
        raise NotImplementedError

    def tune(self):
        assert 0

    # This scaling should undo anything done to the system to make it tractable for the solver.
    def evaluate(self):
        self.action_sequence = self.io.load_trajectory()
        super().evaluate()

    @staticmethod
    def convert_to_env_actions(optimal_control, state):
        return optimal_control

    def convert_to_env_states(self, states):
        return states

    def reset(self):
        self.time = 0

    def get_policy_action(self, observation):
        action = self.scale_action(self.action_sequence[:, self.time], observation)
        assert not np.isnan(action).any()
        # Saturate at the last action.
        if self.time < len(self.action_sequence[0]) - 1:
            self.time += 1
        return action

    @staticmethod
    def scale_action(action, observation):
        raise NotImplementedError

    @staticmethod
    def get_valid_controller_settings():
        return ["casadi_settings", "use_midpoint"]


# For MPC, still do initial optimisation in training phase.
# Only behaviour for get_policy_action is different as this is where the
# current state is updated and the optimal control is rerun.
class MPCOptimalControlTarget(OptimalControlTarget):
    def __init__(self, trial_settings):
        super().__init__(trial_settings)
        self.control_horizon = self.controller_settings.get("control_horizon", 4)
        self.next_offset = self.control_horizon

        # Only store the action sequence for the latest iteration so need
        # to record the offset between the indices of the matrix and the
        # absolute time.
        self.time_offset = 0

        # Store the sequence generated in training so it can be brought out at the
        # start of every run.
        self.reset_action_sequence = None

    def train(self):
        super().train()
        self.io.save_oc_dump(self.optimal_control)

    def evaluate(self):
        self.reset_action_sequence = self.io.load_trajectory()
        self.optimal_control = self.io.load_oc_dump()
        self.get_casadi_settings()
        super().evaluate()

    def get_policy_action(self, observation):
        if (self.time % self.control_horizon) == 0:
            self.update_action_sequence(observation)
        action = self.scale_action(self.action_sequence[:, self.time - self.time_offset], observation)
        assert not np.isnan(action).any()
        self.time += 1
        return action

    def update_action_sequence(self, observation):
        if self.time == 0:
            self.time_offset = self.time
            self.action_sequence = self.reset_action_sequence
            self.next_offset = self.control_horizon
            return

        # update initial state and control initialisation.
        initial_state = self.observation_to_initial_state(observation)
        self.casadi_settings["initial_state"] = initial_state
        control_initial_trajectories = np.zeros_like(self.optimal_control)
        # Use the previous optimal control to initialise
        control_initial_trajectories[:, 0:-self.next_offset] \
            = self.optimal_control[:, self.next_offset:]
        # Duplicate the final optimal control value.
        control_initial_trajectories[:, -self.next_offset:] \
            = control_initial_trajectories[:, -self.next_offset - 1][:, None]
        self.casadi_settings["control_initial_trajectories"] = control_initial_trajectories.transpose().tolist()
        self.check_casadi_settings()
        # Can't sensibly check / enforce inequality constraints in this case as the control and
        # state are not aligned.
        optimiser = OptimalControlCollocation(**self.casadi_settings,
                                              initialisation_plot_path=self.io.optimiser_init_plot_path(),
                                              mpc_iteration=True)

        # Check output and stop if optimiser has not converged.
        success, optimal_control, state = optimiser.run()

        if not success:
            print("Optimiser did not converge. Continuing on previous trajectory")
            self.next_offset += self.control_horizon
            return
        print("Completed optimisation.")

        self.time_offset = self.time
        self.next_offset = self.control_horizon
        self.optimal_control = optimal_control
        optimiser.plot(state_line_type=['.']*self.num_states,
                       control_line_type=['-.']*self.num_controls,
                       filepath=self.io.optimiser_plot_path(timestep=self.time))

        self.env_actions = self.convert_to_env_actions(self.optimal_control, state)
        self.plot_env_actions()
        # Save the control trajectories to a file.
        self.io.save_trajectory(self.env_actions)
        self.action_sequence = self.env_actions

    def observation_to_initial_state(self, observation):
        raise NotImplementedError

    def get_valid_controller_settings(self):
        settings = super().get_valid_controller_settings()
        settings += ['control_horizon']
        return settings
