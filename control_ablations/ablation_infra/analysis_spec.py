class AnalysisProperties:
    def __init__(self,
                 target_settings,
                 test_name,
                 display_name,
                 short_display_name,
                 is_passthrough):
        self.target_settings = target_settings
        self.test_name = test_name
        self.display_name = display_name
        self.short_display_name = short_display_name
        self.is_passthrough = is_passthrough

        self.relevant_analysis = {}

    def set_relevant_analysis(self, name, properties):
        self.relevant_analysis[name] = properties

    def get_property_of_analysis(self, analysis_name, property_name):
        return self.relevant_analysis[analysis_name].get(property_name, None)


class AnalysisSpec:
    def __init__(self, baseline_name, baseline_display):
        self.baseline_name = baseline_name
        self.baseline_display = baseline_display
        self.analysis_properties_list = []
        self.common_properties = {}

    def set_common_properties(self, analysis_name, properties):
        self.common_properties[analysis_name] = properties

    def get_common_properties(self, analysis_name):
        return self.common_properties[analysis_name]

    def add_analysis_properties(self, analysis_properties: AnalysisProperties):
        self.analysis_properties_list.append(analysis_properties)

    @property
    def test_name_list(self):
        return [x.test_name for x in self.analysis_properties_list]

    @property
    def test_name_display_name_pairs(self):
        return [(x.test_name, x.display_name) for x in self.analysis_properties_list]

    def get_passthrough_target_settings(self):
        baseline = [x.target_settings for x in self.analysis_properties_list if x.is_passthrough]
        assert len(baseline) == 1
        return baseline[0]

    def get_passthrough_test_name(self):
        baseline = [x.test_name for x in self.analysis_properties_list if x.is_passthrough]
        assert len(baseline) == 1
        return baseline[0]

    def get_passthrough_display_name(self, short=False):
        if short:
            baseline = [x.short_display_name for x in self.analysis_properties_list if x.is_passthrough]
        else:
            baseline = [x.display_name for x in self.analysis_properties_list if x.is_passthrough]
        assert len(baseline) == 1
        return baseline[0]

    def get_relevant_name_pairs(self, analysis_name, short=False):
        if short:
            return [(x.test_name, x.short_display_name)
                    for x in self.analysis_properties_list if analysis_name in x.relevant_analysis]
        else:
            return [(x.test_name, x.display_name)
                    for x in self.analysis_properties_list if analysis_name in x.relevant_analysis]

    def get_relevant_comparison(self, analysis_name, short=False):
        # Generates a pairwise comparison between the valid tests.
        # Makes sure that the baseline test is first (which makes other logic easier for comparing baseline)
        pairs = self.get_relevant_name_pairs(analysis_name, short)
        baseline_test_name = self.get_passthrough_test_name()
        baseline_index = [index for index, name_pair in enumerate(pairs) if name_pair[0] == baseline_test_name]
        assert (len(baseline_index) == 1)
        pairs.insert(0, pairs.pop(baseline_index[0]))
        out = []
        for index, (test_name1, display_name1) in enumerate(pairs):
            for test_name2, display_name2 in pairs[index + 1:]:
                out.append((test_name1, display_name1, test_name2, display_name2))
        return out

    def get_test_name_from_display_name(self, display_name):
        test_names = [x.test_name for x in self.analysis_properties_list if x.display_name == display_name]
        assert len(test_names) == 1
        return test_names[0]

    def get_display_name_from_test_name(self, test_name):
        display_names = [x.display_name for x in self.analysis_properties_list if x.test_name == test_name]
        assert len(display_names) == 1
        return display_names[0]

    def get_target_settings(self, test_name):
        matching_settings = [x.target_settings for x in self.analysis_properties_list if x.test_name == test_name]
        assert len(matching_settings) == 1
        return matching_settings[0]

    def get_test_name_list_relevant_to_analysis(self, analysis_name):
        return [x.test_name for x in self.analysis_properties_list if analysis_name in x.relevant_analysis]

    def get_test_property(self, test_name, analysis_name, analysis_property):
        aps = [x for x in self.analysis_properties_list if x.test_name == test_name]
        assert len(aps) == 1
        property_contents = aps[0].get_property_of_analysis(analysis_name, analysis_property)
        return property_contents
