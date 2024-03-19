from control_ablations.ablation_infra import AblationStudy, AnalysisSpec, AnalysisProperties, AblationAnalyser, SweepStudy, \
    NormalisedSweepStudy


class AblationStudyWithAnalysis(AblationStudy):
    def __init__(self,
                 target_factory,
                 baseline,
                 baseline_mod,
                 ablation_list,
                 common_analysis_properties,
                 relevant_analysis,
                 apply_overrides,
                 plotter_block_class_list,
                 passthrough_label='baseline'):

        super().__init__(target_factory,
                         baseline,
                         baseline_mod,
                         ablation_list)
        analysis_spec = AnalysisSpec(baseline.__name__,
                                     self.get_baseline_display())
        for analysis_type, settings in common_analysis_properties.items():
            analysis_spec.set_common_properties(analysis_type, settings)
        for idx, a in enumerate(ablation_list):
            is_baseline = a.__name__ == passthrough_label
            ap = AnalysisProperties(self.target_settings_from_base_and_ablate(a),
                                    self.test_name_from_base_and_ablate(a),
                                    self.display_name_from_base_and_ablate(a, short=False),
                                    self.display_name_from_base_and_ablate(a, short=True),
                                    is_baseline
                                    )
            for analysis_type, applicable_settings_tuple in relevant_analysis.items():
                applicable_list, raw_settings = applicable_settings_tuple
                if a in applicable_list:
                    settings = apply_overrides(raw_settings, a, analysis_type)
                    ap.set_relevant_analysis(analysis_type, settings)
            analysis_spec.add_analysis_properties(ap)

        self.analyser = AblationAnalyser(analysis_spec, plotter_block_class_list)

    def get_baseline_display(self):
        target_settings = self.baseline()
        return target_settings.baseline_display_name

    def display_name_from_base_and_ablate(self, ablate, short):
        target_settings = self.target_settings_from_base_and_ablate(ablate)
        return target_settings.get_display_name(short)

    def display_name_from_base_and_simple(self, ablate, short):
        target_settings = self.target_settings_from_base_and_simple(ablate)
        return target_settings.get_display_name(short)

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError


class SweepStudyWithAnalysis(SweepStudy):
    def __init__(self,
                 target_factory,
                 baseline,
                 baseline_mod,
                 sweep_function,
                 sweep_points,
                 sweep_point_for_baseline,
                 common_analysis_properties,
                 relevant_analysis,
                 apply_overrides,
                 plotter_block_class_list
                 ):
        super().__init__(target_factory,
                         baseline,
                         baseline_mod,
                         sweep_function,
                         sweep_points)

        analysis_spec = AnalysisSpec(self.baseline.__name__, self.get_baseline_display())
        for analysis_type, settings in common_analysis_properties.items():
            analysis_spec.set_common_properties(analysis_type, settings)

        for n in sweep_points:
            target_settings = self.target_settings_from_sweep(n)
            is_baseline = n == sweep_point_for_baseline
            ap = AnalysisProperties(target_settings,
                                    self.experiment_name_from_sweep(n),
                                    self.display_name_from_base_and_sweep_point(n, short=False),
                                    self.display_name_from_base_and_sweep_point(n, short=True),
                                    is_baseline)
            for analysis_type, applicable_settings_tuple in relevant_analysis.items():
                applicable_list, raw_settings = applicable_settings_tuple
                if n in applicable_list:
                    settings = apply_overrides(raw_settings, n, analysis_type)
                    ap.set_relevant_analysis(analysis_type, settings)
            analysis_spec.add_analysis_properties(ap)
        self.analyser = AblationAnalyser(analysis_spec, plotter_block_class_list)

    def get_baseline_display(self):
        target_settings = self.baseline()
        return target_settings.baseline_display_name

    def display_name_from_base_and_sweep_point(self, sweep_point, short):
        target_settings = self.target_settings_from_sweep(sweep_point)
        return target_settings.get_display_name(short)

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError


class NormalisedSweepStudyWithAnalysis(NormalisedSweepStudy):
    # Changes reward and changes interface refer to the sweep. It is assumed that the
    # control and test ablations change neither.
    def __init__(self,
                 target_factory,
                 baseline,
                 baseline_mod,
                 control_ablation,
                 test_ablation,
                 sweep_function,
                 sweep_points,
                 sweep_point_for_baseline,
                 common_analysis_properties,
                 relevant_analysis,
                 apply_overrides,
                 plotter_block_class_list):
        super().__init__(target_factory,
                         baseline,
                         baseline_mod,
                         control_ablation,
                         test_ablation,
                         sweep_function,
                         sweep_points)
        analysis_spec = AnalysisSpec(self.baseline.__name__, self.get_baseline_display())
        for analysis_type, settings in common_analysis_properties.items():
            analysis_spec.set_common_properties(analysis_type, settings)
        for control_or_test in ["control", "test"]:
            if control_or_test == "control":
                ablation = control_ablation
            else:
                ablation = test_ablation
            for n in sweep_points:
                target_settings = self.target_settings_from_sweep_and_ablate(n, ablation)
                is_baseline = (n == sweep_point_for_baseline) and (ablation == control_ablation)
                ap = AnalysisProperties(target_settings,
                                        self.experiment_name_from_sweep_and_ablate(n, ablation),
                                        self.display_name_from_base_sweep_point_ablation(n, ablation, short=False),
                                        self.display_name_from_base_sweep_point_ablation(n, ablation, short=True),
                                        is_baseline)
                # Some properties are relevant for anything to do with a normalised sweep
                normalise_sweep_settings = {"control_or_test": control_or_test,
                                            "sweep_point": n}
                ap.set_relevant_analysis("normalised_sweep", normalise_sweep_settings)
                for analysis_type, applicable_settings_tuple in relevant_analysis.items():
                    applicable_list, raw_settings = applicable_settings_tuple
                    if n in applicable_list:
                        settings = apply_overrides(raw_settings, n, analysis_type)
                        ap.set_relevant_analysis(analysis_type, settings)

                analysis_spec.add_analysis_properties(ap)
        self.analyser = AblationAnalyser(analysis_spec, plotter_block_class_list)

    def display_name_from_base_sweep_point_ablation(self, sweep_point, ablation, short):
        target_settings = self.target_settings_from_sweep_and_ablate(sweep_point, ablation)
        return target_settings.get_display_name(short)

    def get_baseline_display(self):
        target_settings = self.baseline()
        return target_settings.baseline_display_name

    def get_per_agent_task_duration_in_hours(self):
        raise NotImplementedError
