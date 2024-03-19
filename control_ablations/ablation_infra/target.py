from control_ablations.ablation_infra.trial_settings import CEPATrialSettings, CEPATargetSettings
from control_ablations.ablation_infra import PerIterationIO


class Target:
    def __init__(self, run_settings):
        self.run_settings = run_settings

    def run(self):
        if self.run_settings.do_tuning():
            self.tune()
        if self.run_settings.do_training():
            self.train()
        if self.run_settings.do_eval():
            self.evaluate()
        if self.run_settings.do_plotting():
            self.plot()

    def tune(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


class CEPATarget(Target):

    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings.run_settings)
        self.test_name = trial_settings.test_name

        target_settings = trial_settings.target_settings
        assert (isinstance(target_settings, CEPATargetSettings))
        assert (set(target_settings.controller.keys()) == {"settings", "type"})
        self.controller_settings = target_settings.controller["settings"]
        self.sim_setup = target_settings.sim_setup
        self.env_params = target_settings.env_params
        self.make_env = target_settings.make_env
        self.plot_display_name = target_settings.get_display_name(short=False)
        self.io = PerIterationIO(self.test_name)

    def tune(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def check_settings_are_valid(self):
        assert (callable(self.make_env))
        valid_controller_settings = self.get_valid_controller_settings()
        for key in self.controller_settings:
            if key not in valid_controller_settings:
                print(f"{key} is not a valid controller setting for this target")
                assert key in valid_controller_settings

    @staticmethod
    def get_valid_controller_settings():
        raise NotImplementedError()

    def get_seed_offset_and_logdir_root(self):
        eval_idx = self.run_settings.eval_settings.get("eval_idx", None)
        seed_offset = self.run_settings.eval_settings.get("seed_offset", 0)
        # If the evaluation has been split into parts, need to sort out seed offsets and logging
        if eval_idx is not None:
            seed_offset += eval_idx * 1000
            logdir_root = self.io.get_split_eval_dir(eval_idx)
        else:
            logdir_root = self.run_settings.eval_settings.get("logdir_root", None)
        return seed_offset, logdir_root
