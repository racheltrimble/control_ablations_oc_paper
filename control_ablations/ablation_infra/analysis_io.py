import tarfile
from pathlib import Path

from control_ablations.ablation_infra.base_io import BaseIO


class AnalysisIO(BaseIO):
    def __init__(self, summary_name, experiment_name_list):
        self.summary_name = summary_name
        self.experiment_name_list = experiment_name_list

    def get_analysis_for_experiment(self, experiment_name):
        return self.get_analysis_root() / Path(experiment_name)

    def get_analysis_summary_dir(self):
        analysis_dir = self.get_analysis_root() / Path(self.summary_name)
        if not analysis_dir.exists():
            analysis_dir.mkdir()

        return analysis_dir

    # Intended to be used to store the git revision of the code used to generate the results.
    def get_tag_file_path(self):
        analysis_dir = self.get_analysis_summary_dir()
        return analysis_dir / Path("tag.txt")

    def get_tar_path(self):
        return self.get_output_root() / Path(self.summary_name + ".tar")

    def filelist_and_summary_dir_to_tar(self, extra_file_list, tar_data):
        tar_name = self.get_tar_path()
        if Path(tar_name).exists():
            Path(tar_name).unlink()

        # From the overall results folder, get everything (includes learning histograms at different steps + latex)
        glob_out = self.get_analysis_summary_dir().glob("*")
        extra_file_list += [(str(f), f.parts[-1]) for f in glob_out]

        tar_name = self.get_tar_path()
        with tarfile.open(str(tar_name), mode='x:gz') as tar:
            for file in extra_file_list:
                print("Adding " + file[0] + " as " + file[1])
                tar.add(file[0], arcname=file[1])
            if tar_data:
                tar.add(self.get_data_dir())

    # Expecting this to be overridden in the IO class when extra files are required.
    @staticmethod
    def get_extra_files_for_tar_per_experiment(_experiment_name, _folder_name):
        return []
