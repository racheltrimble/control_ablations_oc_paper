import csv
from pathlib import Path
from control_ablations import config

class BaseIO:
    @staticmethod
    def get_root_dir():
        return config.logging_root_path

    def get_data_dir(self):
        root_dir = self.get_root_dir()
        return root_dir / Path("data")

    def get_analysis_root(self):
        root_dir = self.get_root_dir()
        return root_dir / Path("analysis")

    def get_output_root(self):
        root_dir = self.get_root_dir()
        return root_dir / Path("output")


class PerTargetIO(BaseIO):

    def __init__(self, test_name):
        self.test_name = test_name

    def get_overall_logging_dir(self):
        return self.get_data_dir() / Path(self.test_name)

    def get_analysis_dir(self):
        analysis_dir = self.get_analysis_root() / Path(self.test_name)
        if not analysis_dir.exists():
            analysis_dir.mkdir()
        return analysis_dir

    def get_test_iterations(self):
        test_dir = self.get_overall_logging_dir()
        if test_dir.exists():
            sub_dir_list = [x for x in test_dir.iterdir() if (x.is_dir())]
            return sub_dir_list
        else:
            return []


class PerIterationIO(PerTargetIO):
    def __init__(self, test_name):
        super().__init__(test_name)

        self.iteration = None

    def set_iteration(self, iteration):
        self.iteration = iteration

    def get_iteration_dir(self):
        assert self.iteration is not None
        return self.get_overall_logging_dir() / Path(str(self.iteration))

    def get_eval_dir(self, logdir_root=None):
        if logdir_root is None:
            log_dir = self.get_iteration_dir()
            return log_dir / Path("eval")
        else:
            log_dir = logdir_root / Path(self.test_name) / Path(str(self.iteration))
            return log_dir / Path("eval")

    def get_reward_path(self, logdir_root=None):
        return self.get_eval_dir(logdir_root) / Path("rewards.csv")

    def write_reward_file(self, rewards, logdir_root=None):
        path = self.get_reward_path(logdir_root)
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(rewards)

    def read_reward_file(self, logdir_root=None):
        path = self.get_reward_path(logdir_root)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                list_o_strings = list(row)
        return [float(a) for a in list_o_strings]

    def dump_training_time(self, training_time):
        filename = self.get_iteration_dir() / Path("training_time.txt")
        with open(filename, 'w') as f:
            f.write(training_time)

    def get_split_eval_dir(self, eval_idx):
        return self.get_data_dir() / Path(f"splits{eval_idx}")
