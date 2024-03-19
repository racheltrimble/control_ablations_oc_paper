import numpy as np

from control_ablations.ablation_infra import PerIterationIO
from pathlib import Path


# There is no real concept of iterations but it is more consistent to implement
# with everything as iteration 0 rather than making a different structure for these
# types of tests e.g. Allows reuse of eval code.
class OCIO(PerIterationIO):
    def __init__(self, test_name):
        super().__init__(test_name)
        self.set_iteration(0)

    def make_training_file_structure(self):
        # Create overall logging directory
        per_trial_dir = self.get_iteration_dir()
        print("Creating overall logging dir:", str(per_trial_dir))
        per_trial_dir.mkdir(exist_ok=True, parents=True)

    def save_trajectory(self, trajectory):
        np.savetxt(self.control_trajectory_path(), trajectory)

    def load_trajectory(self):
        trajectory = np.genfromtxt(self.control_trajectory_path())
        return trajectory

    def save_state_trajectory(self, trajectory):
        np.savetxt(self.state_trajectory_path(), trajectory)

    def load_state_trajectory(self):
        trajectory = np.genfromtxt(self.state_trajectory_path())
        return trajectory

    def control_trajectory_path(self):
        return self.get_iteration_dir() / Path("optimal_trajectory.txt")

    def state_trajectory_path(self):
        return self.get_iteration_dir() / Path("optimal_state_trajectory.txt")

    def save_oc_dump(self, trajectory):
        np.savetxt(self.oc_dump_path(), trajectory)

    def load_oc_dump(self):
        trajectory = np.genfromtxt(self.oc_dump_path())
        return trajectory

    def oc_dump_path(self):
        return self.get_iteration_dir() / Path("oc_raw_dump.txt")

    def optimiser_plot_path(self, timestep=0):
        return self.get_iteration_dir() / Path(f"optimiser_plot_{timestep}.png")

    def optimiser_init_plot_path(self):
        return self.get_iteration_dir() / Path("optimiser_init_plot.png")

    def env_plot_path(self, timestep=0):
        return self.get_iteration_dir() / Path(f"env_instructions_{timestep}.png")
