import os
from pathlib import Path
from shutil import rmtree
from subprocess import run

from control_ablations.ablation_infra import BaseIO


class Runner:
    def __init__(self, baseline_name):
        self.baseline_name = baseline_name
        self.io = BaseIO()

    def check_clean(self):
        clean = True
        clean &= len(self.folder_contents_without_hidden(self.io.get_data_dir())) == 0
        clean &= len(self.folder_contents_without_hidden(self.io.get_analysis_root())) == 0
        clean &= len(self.folder_contents_without_hidden(self.io.get_output_root())) == 0
        if not clean:
            print("Runner requested to check directories clean before test run and test failed")
            assert 0
        return clean

    @staticmethod
    def folder_contents_without_hidden(folder):
        raw = os.listdir(folder)
        return [file for file in raw if not file.startswith(".")]

    def get_setup_file_dir(self):
        setup_dir = self.io.get_root_dir() / Path("setup")
        if not setup_dir.exists():
            setup_dir.mkdir()

        return setup_dir

    def run_per_agent_tasks(self, ts_list, time_limit_in_hours):
        file_count = self.gen_setup_scripts_for_per_agent_tasks(ts_list)
        self.run_per_agent_tasks_from_files(file_count, time_limit_in_hours)

    def run_per_agent_tasks_from_files(self, file_count, time_limit_in_hours):
        raise NotImplementedError

    # Generates the scripting to run an ablation as per run_ablations_from_list
    def gen_setup_scripts_for_per_agent_tasks(self, ts_list):
        # Clear the setup file directory - these are autogenned so don't need to stick around
        rmtree(self.get_setup_file_dir())

        file_count = 0
        for trial in ts_list:
            filename = self.get_setup_file_dir() / Path(self.baseline_name + str(file_count) + ".yaml")
            trial.write_to_file(filename)
            file_count += 1
        print("Generated ", file_count, " files for ablation run.")
        return file_count

    # Some runners don't need a finalise step
    def finalise(self):
        pass

    def run_analyser(self, analyser, analysis_settings):
        raise NotImplementedError


class LocalRunner(Runner):
    def __init__(self, baseline_name, target_factory):
        super().__init__(baseline_name)
        self.target_factory = target_factory

    def run_per_agent_tasks_from_files(self, _1, _2):
        filename_list = os.listdir(self.get_setup_file_dir())
        # Run baseline and then each of the ablations
        for setup in filename_list:
            full_name = self.get_setup_file_dir() / Path(str(setup))
            target = self.target_factory.get_target_from_file(full_name)
            target.run()

    def run_analyser(self, analyser, analysis_settings):
        analyser.run(analysis_settings)


# Runs jobs locally but does it by submitting individual commands to the terminal
# Used when memory overflow is risky for larger jobs.
class LocalSeparatedRunner(LocalRunner):
    def __init__(self, baseline_name):
        # Not passing target factory as only needed for main runs (not used in this approach)
        super().__init__(baseline_name, None)

    def run_per_agent_tasks_from_files(self, _1, _2):
        filename_list = os.listdir(self.get_setup_file_dir())
        # Local runner can do big jobs in bits while maintaining ordering.
        filename_list.sort()
        # Run baseline and then each of the ablations
        target_py_path = (self.io.get_root_dir() / Path("plant_disease_model") / Path("control")
                          / Path("target_factory.py"))
        for setup in filename_list:
            full_name = self.get_setup_file_dir() / Path(str(setup))
            run(["python3", target_py_path.as_posix(), full_name])

