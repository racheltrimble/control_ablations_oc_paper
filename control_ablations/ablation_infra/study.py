from control_ablations.ablation_infra import RunSettings, TrialSettings
from control_ablations.ablation_infra.runner import LocalRunner, LocalSeparatedRunner


class Study:
    def __init__(self, target_factory, baseline, baseline_mod):
        if baseline_mod is None:
            self.baseline = baseline
        else:
            def baseline_modded():
                target_settings = baseline()
                return baseline_mod(target_settings)

            self.baseline = baseline_modded
        self.study_name = baseline.__name__
        self.target_factory = target_factory

        # Gets set as kwargs on run
        self.ablation_config = None
        self.repeats = None
        self.tar_data = None
        self.runner = None
        self.analyser = not_implemented_field

    def run_from_template(self, template_name):
        ablation_config = {}
        if template_name == "cluster":
            ablation_config["for_cluster"] = True
            ablation_config["check_clean"] = True
        elif template_name == "full":
            ablation_config = {}
        elif template_name == "analysis_only":
            ablation_config["run_training"] = False
            ablation_config["generate_examples"] = False
            ablation_config["run_plotting"] = False
        else:
            print("Unrecognised template name passed to ablation study")
            assert 0
        self.run(**ablation_config)

    def run_from_command_line(self, argv):
        assert len(argv) == 2
        self.run_from_template(argv[1])

    def run(self, **kwargs):
        print("Running ablation study")
        self.ablation_config = kwargs
        self.check_ablation_config()

        # Pull out parameter style configurations here
        # Switches to enable different phases are handled below.
        self.repeats = self.ablation_config.get("repeats", 10)

        eval_run_settings = self.ablation_config.get("eval_run_settings", {})
        plotting_run_settings = self.ablation_config.get("plotting_run_settings", {})

        check_clean = self.ablation_config.get("check_clean", False)
        run_training = self.ablation_config.get("run_training", True)
        generate_examples = self.ablation_config.get("generate_examples", True)
        run_plotting = self.ablation_config.get("run_plotting", True)
        for_cluster = self.ablation_config.get("for_cluster", False)
        run_via_command_line = self.ablation_config.get("run_via_command_line", False)
        split_eval_runs_into_groups_of = self.ablation_config.get("split_eval_runs_into_groups_of", None)

        big_analyser_settings = {
            "run_analysis": self.ablation_config.get("run_analysis", True),
            "generate_tarball": self.ablation_config.get("generate_tarball", True),
            "perf_comparison_repeats": self.ablation_config.get("perf_comparison_repeats", 1_000),
            "tar_data": self.ablation_config.get("tar_data", True)
        }
        analyser_settings = {**big_analyser_settings, **self.ablation_config.get("analyser_settings", {})}

        # Not supported in this version.
        assert(not for_cluster)
        if run_via_command_line:
            self.runner = LocalSeparatedRunner(self.study_name)
        else:
            self.runner = LocalRunner(self.study_name, self.target_factory)

        run_settings = RunSettings()

        iterations = list(range(self.repeats))
        if run_training:
            run_settings.add_training(iterations=iterations)
        if generate_examples:
            run_settings.add_eval(iterations=iterations,
                                  **eval_run_settings)
        if run_plotting:
            run_settings.add_plotting(iterations=iterations,
                                      **plotting_run_settings)
        if check_clean:
            self.runner.check_clean()
        if run_training or generate_examples or run_plotting:
            ts_list = self.generate_trials(run_settings,
                                           split_for_cluster=True,
                                           split_eval_runs_into_groups_of=split_eval_runs_into_groups_of)
            time_limit_in_hours = self.get_per_agent_task_duration_in_hours()
            self.runner.run_per_agent_tasks(ts_list, time_limit_in_hours)
        if analyser_settings["run_analysis"] or analyser_settings["generate_tarball"]:
            self.runner.run_analyser(self.analyser, analyser_settings)
        self.runner.finalise()

    def check_ablation_config(self):
        allowed_keys = ["repeats",
                        "eval_run_settings",
                        "plotting_run_settings",
                        "check_clean",
                        "run_training",
                        "generate_examples",
                        "run_plotting",
                        "for_cluster",
                        "run_via_command_line",
                        "split_eval_runs_into_groups_of",
                        "run_analysis",
                        "generate_tarball",
                        "perf_comparison_repeats",
                        "tar_data",
                        "analyser_settings"
                        ]
        for key in self.ablation_config:
            assert (key in allowed_keys)

    def generate_trials(self, run_settings, split_for_cluster, split_eval_runs_into_groups_of):
        raise NotImplementedError

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError

    @staticmethod
    def post_process_ts_to_list(ts, split_for_cluster, split_eval_runs_into_groups_of):
        # Split into separate training iterations
        if split_for_cluster:
            split_list = ts.split_for_cluster()
        else:
            split_list = [ts]
        # If required, split the evaluation into chunks.
        if split_eval_runs_into_groups_of is not None:
            out = []
            for trial in split_list:
                out += trial.split_eval_runs(split_eval_runs_into_groups_of)
        else:
            out = split_list
        return out


@property
def not_implemented_field():
    raise NotImplementedError("Subclasses should implement this!")


# Series of trials based on a baseline and a list of functions that
# alter the baseline.
class AblationStudy(Study):
    def __init__(self, target_factory, baseline, baseline_mod, ablation_list):
        super().__init__(target_factory, baseline, baseline_mod)
        self.ablation_list = ablation_list

    def target_settings_from_base_and_simple(self, control):
        target_settings = self.baseline()
        target_settings = control(target_settings)
        return target_settings

    def test_name_from_base_and_simple(self, control):
        return self.baseline.__name__ + "_" + control.__name__

    def generate_trials(self, run_settings, split_for_cluster, split_eval_runs_into_groups_of):
        ts_list = []
        for ablate in self.ablation_list:
            target_settings = self.target_settings_from_base_and_ablate(ablate)
            test_name = self.test_name_from_base_and_ablate(ablate)
            ts = TrialSettings(test_name, run_settings, target_settings)
            split_list = self.post_process_ts_to_list(ts, split_for_cluster, split_eval_runs_into_groups_of)
            ts_list += split_list
        return ts_list

    def target_settings_from_base_and_ablate(self, ablate):
        target_settings = self.baseline()
        target_settings = ablate(target_settings)
        return target_settings

    def test_name_from_base_and_ablate(self, ablate):
        return self.baseline.__name__ + "_" + ablate.__name__

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError


# Series of trials based on a baseline, a function that alters the baseline
# and a list of parameters passed to that function for each of the trials.
class SweepStudy(Study):
    def __init__(self, target_factory, baseline, baseline_mod, sweep_function, sweep_points):
        super().__init__(target_factory, baseline, baseline_mod)
        self.sweep_function = sweep_function
        self.sweep_points = sweep_points

    def generate_trials(self, run_settings, split_for_cluster, split_eval_runs_into_groups_of):
        ts_list = []
        for sweep_point in self.sweep_points:
            target_settings = self.target_settings_from_sweep(sweep_point)
            test_name = self.experiment_name_from_sweep(sweep_point)
            ts = TrialSettings(test_name, run_settings, target_settings)
            split_list = self.post_process_ts_to_list(ts, split_for_cluster, split_eval_runs_into_groups_of)
            ts_list += split_list
        return ts_list

    def target_settings_from_sweep(self, sweep_point):
        target_settings = self.baseline()
        target_settings = self.sweep_function(target_settings, sweep_point)
        return target_settings

    def experiment_name_from_sweep(self, sweep_point):
        return self.baseline.__name__ + "_" + self.sweep_function.__name__ + "_" + str(sweep_point)

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError


# As per sweep study but run twice - once with a control preprocessing ablation function and once with
# a test ablation function. This allows the ablated sweep to be normalised against the control.
class NormalisedSweepStudy(Study):
    def __init__(self,
                 target_factory,
                 baseline,
                 baseline_mod,
                 control_ablation,
                 test_ablation,
                 sweep_function,
                 sweep_points):
        super().__init__(target_factory, baseline, baseline_mod)
        self.control_ablation = control_ablation
        self.test_ablation = test_ablation
        self.sweep_function = sweep_function
        self.sweep_points = sweep_points

    def generate_trials(self, run_settings, split_for_cluster, split_eval_runs_into_groups_of):
        ts_list = []
        for ablation in [self.control_ablation, self.test_ablation]:
            for sweep_point in self.sweep_points:
                target_settings = self.target_settings_from_sweep_and_ablate(sweep_point, ablation)
                test_name = self.experiment_name_from_sweep_and_ablate(sweep_point, ablation)
                ts = TrialSettings(test_name, run_settings, target_settings)
                split_list = self.post_process_ts_to_list(ts, split_for_cluster, split_eval_runs_into_groups_of)
                ts_list += split_list
        return ts_list

    def target_settings_from_sweep_and_ablate(self, sweep_point, ablation):
        target_settings = self.baseline()
        target_settings = ablation(target_settings)
        target_settings = self.sweep_function(target_settings, sweep_point)
        return target_settings

    def experiment_name_from_sweep_and_ablate(self, sweep_point, ablation):
        baseline = self.baseline.__name__
        ablation_name = ablation.__name__
        sweep = self.sweep_function.__name__
        return baseline + "_" + ablation_name + "_" + sweep + "_" + str(sweep_point)

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError
