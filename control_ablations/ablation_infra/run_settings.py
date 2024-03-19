from copy import deepcopy
from pathlib import Path
import math


class RunSettings:
    def __init__(self):
        self.tune_settings = {}
        self.train_settings = {}
        self.eval_settings = {}
        self.plot_settings = {}

    @classmethod
    def from_yaml(cls, file_contents):
        out = cls()
        out.tune_settings = file_contents["tune_settings"]
        out.train_settings = file_contents["train_settings"]
        out.eval_settings = cls.eval_settings_from_yaml(file_contents["eval_settings"])
        out.plot_settings = file_contents["plot_settings"]
        return out

    def to_dict(self):
        out = {"tune_settings": self.tune_settings,
               "train_settings": self.train_settings,
               "eval_settings": self.eval_settings_to_dict(),
               "plot_settings": self.plot_settings.copy()}
        return out

    def eval_settings_to_dict(self):
        val = self.eval_settings.copy()
        if "logdir_root" in val:
            val["logdir_root"] = val["logdir_root"].as_posix()
        return val

    @staticmethod
    def eval_settings_from_yaml(val):
        if "logdir_root" in val:
            val["logdir_root"] = Path(val["logdir_root"])
        return val

    def copy(self):
        new_me = RunSettings()
        new_me.tune_settings = deepcopy(self.tune_settings)
        new_me.train_settings = deepcopy(self.train_settings)
        new_me.eval_settings = deepcopy(self.eval_settings)
        new_me.plot_settings = deepcopy(self.plot_settings)

        return new_me

    def set_iterations(self, iterations):
        if self.do_training():
            self.train_settings["iterations"] = self.convert_to_list(iterations)
        if self.do_eval():
            self.eval_settings["iterations"] = self.convert_to_list(iterations)
        if self.do_plotting():
            self.plot_settings["iterations"] = self.convert_to_list(iterations)

    def set_eval_repeats(self, allocation):
        self.eval_settings["example_plot_repeats"] = allocation

    def set_eval_idx(self, idx):
        self.eval_settings["eval_idx"] = idx

    def add_tuning(self):
        self.tune_settings["on"] = True

    # Can be a list of iteration numbers or an int specifying a particular iteration.
    # Note must use a combination of list and range to specify multiple runs.
    def add_training(self, **kwargs):
        self.train_settings = kwargs
        self.train_settings["iterations"] = self.convert_to_list(self.train_settings["iterations"])

    def add_eval(self, **kwargs):
        self.eval_settings = kwargs
        self.eval_settings["iterations"] = self.convert_to_list(self.eval_settings["iterations"])

    def add_plotting(self, **kwargs):
        self.plot_settings = kwargs
        self.plot_settings["iterations"] = self.convert_to_list(self.plot_settings["iterations"])

    @staticmethod
    def convert_to_list(iterations):
        if isinstance(iterations, int):
            iter_val = [iterations]
        elif isinstance(iterations, list):
            iter_val = iterations
        else:
            print("Unexpected format passed to RunSettings: ", iterations)
            assert 0
        return iter_val

    def do_tuning(self):
        return self.tune_settings.get("on", False)

    def do_training(self):
        return len(self.train_settings) > 0

    def do_eval(self):
        return len(self.eval_settings) > 0

    def do_plotting(self):
        return len(self.plot_settings) > 0

    def split_for_cluster(self):
        # Only supporting this for run settings that are all the same for now.
        template_set = None
        if self.do_training():
            train_set = set(self.train_settings['iterations'])
            template_set = train_set
        if self.do_eval():
            eval_set = set(self.eval_settings['iterations'])
            if template_set is not None:
                assert (template_set == eval_set)
            template_set = eval_set
        if self.do_plotting():
            plot_set = set(self.plot_settings['iterations'])
            if template_set is not None:
                assert (template_set == plot_set)
            template_set = plot_set
        # Can end up with an empty set e.g. if only doing training and training is not valid
        # for this ablation.
        if template_set is None:
            print("Warning - combination of run settings and ablation has resulted in no valid runs.")
            return []

        split_list = []
        for i in template_set:
            new_rs = self.copy()
            new_rs.set_iterations(i)
            split_list.append(new_rs)
        return split_list

    def split_eval_runs(self, split_eval_runs_into_groups_of):
        # Only supported for single iteration evals for local runs for now.
        assert (len(self.eval_settings["iterations"]) == 1)
        assert (split_eval_runs_into_groups_of is not None)

        repeats = self.eval_settings.get("example_plot_repeats", 100)
        number_of_splits = repeats / split_eval_runs_into_groups_of
        number_of_splits_lower = math.floor(number_of_splits)
        number_of_splits_upper = math.ceil(number_of_splits)
        chunked_allocations = [split_eval_runs_into_groups_of] * number_of_splits_lower
        if number_of_splits_lower != number_of_splits_upper:
            chunked_allocations += [repeats - split_eval_runs_into_groups_of*number_of_splits_lower]

        assert (sum(chunked_allocations) == repeats)

        split_list = []
        for idx, allocation in enumerate(chunked_allocations):
            new_rs = self.copy()
            if idx != 0:
                self.plot_settings = {}
                self.train_settings = {}
            new_rs.set_eval_repeats(allocation)
            new_rs.set_eval_idx(idx)
            new_rs.plot_settings = {}
            split_list.append(new_rs)
        return split_list

    def apply_limits(self, tuning_valid=True, training_valid=True, multiple_evals=True):
        # Note - this sets values in train_settings so needs to come above the train_settings overwrite.
        if not multiple_evals:
            if self.train_settings != {}:
                self.train_settings["iterations"] = [0]
            if self.eval_settings != {}:
                self.eval_settings["iterations"] = [0]
            if self.plot_settings != {}:
                self.plot_settings["iterations"] = [0]
        if not tuning_valid:
            self.tune_settings = {}
        if not training_valid:
            self.train_settings = {}
