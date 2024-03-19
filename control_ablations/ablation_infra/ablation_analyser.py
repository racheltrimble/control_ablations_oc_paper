from control_ablations.ablation_infra.analysis_io import AnalysisIO

class AblationAnalyser:
    def __init__(self, analysis_spec, plotter_block_class_list):
        self.analysis_spec = analysis_spec
        self.experiment_name_list = analysis_spec.test_name_list
        self.analyser_settings = None
        self.io = AnalysisIO(self.analysis_spec.baseline_name, self.experiment_name_list)
        self.plotter_block_class_list = plotter_block_class_list

    def run(self, analyser_settings):
        self.analyser_settings = analyser_settings
        if analyser_settings["run_analysis"]:
            self.run_ablation_analysis()
        if analyser_settings["generate_tarball"]:
            self.bundle_results(analyser_settings["tar_data"])

    def run_ablation_analysis(self):
        for block_class in self.plotter_block_class_list:
            if self.analyser_settings.get(block_class.get_trigger(), True):
                block = block_class(self.analysis_spec,
                                    self.io,
                                    self.analyser_settings
                                    )
                block.plot()


    def bundle_results(self, tar_data: bool):
        extra_file_list = []
        # From each test folder get learning curves and the performance plots for an example agent.
        for experiment_name in self.io.experiment_name_list:
            folder_name = self.io.get_analysis_for_experiment(experiment_name)
            # Expecting this to be overridden in the IO class when extra files are required.
            extra_file_list += self.io.get_extra_files_for_tar_per_experiment(experiment_name, folder_name)
        self.io.filelist_and_summary_dir_to_tar(extra_file_list, tar_data)


class BasePlotBlock:
    def plot(self):
        raise NotImplementedError

    @staticmethod
    def get_trigger():
        raise NotImplementedError
