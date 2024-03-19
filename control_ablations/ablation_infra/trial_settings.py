import yaml

from control_ablations.ablation_infra import RunSettings


class TargetSettings:
    def from_dict(self, dictionary):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def get_target_type(self):
        raise NotImplementedError

    def get_run_limits(self):
        raise NotImplementedError


class TrialSettings:
    def __init__(self,
                 test_name: str,
                 run_settings: RunSettings,
                 target_settings: TargetSettings):
        self.test_name = test_name
        # Need to make a copy so the application of limits doesn't affect other runs.
        self.run_settings = run_settings.copy()
        self.target_settings = target_settings
        # Limits are applied on creation so are not included in the YAML IO.
        self.run_settings.apply_limits(**target_settings.get_run_limits())

    def get_target_type(self):
        return self.target_settings.get_target_type()

    @classmethod
    def from_file(cls, setup_file, target_constructor):
        with open(setup_file, "r") as stream:
            try:
                file_contents = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        test_name = file_contents['test_name']
        rs = RunSettings.from_yaml(file_contents["run_settings"])
        ts = target_constructor.from_dict(file_contents["target_settings"])
        return cls(test_name, rs, ts)

    def write_to_file(self, filename):
        contents = {"test_name": self.test_name,
                    "run_settings": self.run_settings.to_dict(),
                    "target_settings": self.target_settings.to_dict()}

        with open(filename, "w") as stream:
            try:
                yaml.dump(contents, stream)
            except yaml.YAMLError as exc:
                print(exc)

    # For use with the cluster, need to write out separate setup files per iteration
    def split_for_cluster(self):
        run_list = self.run_settings.split_for_cluster()
        return self.split_list_from_run_list(run_list)

    def split_list_from_run_list(self, run_list):
        split_list = []
        for new_rs in run_list:
            new_trial = TrialSettings(test_name=self.test_name,
                                      run_settings=new_rs,
                                      target_settings=self.target_settings)
            split_list.append(new_trial)
        return split_list

    def split_eval_runs(self, split_eval_runs_into_groups_of):
        run_list = self.run_settings.split_eval_runs(split_eval_runs_into_groups_of)
        return self.split_list_from_run_list(run_list)


class CEPATargetSettings(TargetSettings):
    def __init__(self, env_params, sim_setup, controller, make_env, baseline_display_name, run_limits=None):
        if run_limits is None:
            self.run_limits = {"tuning_valid": True,
                               "training_valid": True,
                               "multiple_evals": True}
        # Defining the parameters for the trial
        self.env_params = env_params
        self.sim_setup = sim_setup
        self.controller = controller
        self.make_env = make_env
        self.baseline_display_name = baseline_display_name
        self.display_name_addendum = None

    def set_display_name_addendum(self, display_name):
        self.display_name_addendum = display_name

    def get_display_name(self, short):
        if short:
            return self.display_name_addendum
        else:
            return self.baseline_display_name + " " + self.display_name_addendum

    def __eq__(self, other):
        if isinstance(other, CEPATargetSettings):
            out = self.env_params == other.env_params
            out &= self.sim_setup == other.sim_setup
            out &= self.controller == other.controller
            out &= self.make_env == other.make_env

            return out
        return NotImplemented

    @classmethod
    def from_dict(cls, file_contents):
        env_params = cls.env_params_from_dict(file_contents["env_type"],
                                              file_contents["env_params"])
        setup = cls.sim_setup_from_dict(file_contents["setup"])
        controller = cls.controller_from_dict(file_contents["controller"])
        make_env = cls.make_env_from_string(file_contents["make_env"])
        baseline_display_name = file_contents['baseline_display_name']
        display_name_addendum = file_contents['display_name_addendum']
        out = cls(env_params, setup, controller, make_env, baseline_display_name)
        out.set_display_name_addendum(display_name_addendum)
        return out

    @classmethod
    def env_params_from_dict(cls, env_type, env_dict):
        raise NotImplementedError

    @classmethod
    def sim_setup_from_dict(cls, sim_dict):
        raise NotImplementedError

    @classmethod
    def controller_from_dict(cls, controller_dict):
        raise NotImplementedError

    @classmethod
    def make_env_from_string(cls, make_env_str):
        raise NotImplementedError

    def to_dict(self):
        env_type, env_params = self.env_params_to_dict()
        assert (self.display_name_addendum is not None)
        contents = {"setup": self.sim_setup_to_dict(),
                    "env_type": env_type,
                    "env_params": env_params,
                    "controller": self.controller_to_dict(),
                    "make_env": self.make_env_to_string(),
                    "baseline_display_name": self.baseline_display_name,
                    "display_name_addendum": self.display_name_addendum}
        return contents

    def env_params_to_dict(self):
        raise NotImplementedError

    def sim_setup_to_dict(self):
        raise NotImplementedError

    def controller_to_dict(self):
        raise NotImplementedError

    def make_env_to_string(self):
        raise NotImplementedError

    def get_target_type(self):
        return self.controller["type"]

    def get_run_limits(self):
        return self.run_limits


class CEPATrialSettings(TrialSettings):
    def __init__(self,
                 test_name: str,
                 run_settings: RunSettings,
                 target_settings: CEPATargetSettings):
        super().__init__(test_name, run_settings, target_settings)
