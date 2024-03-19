from control_ablations.ablation_infra import CEPATrialSettings, CEPATarget


class NoLearningControlTarget(CEPATarget):
    def tune(self):
        print("No tuning possible for fixed controller")

    def train(self):
        print("No training possible for fixed controller")

    def get_policy_action(self, observation):
        raise NotImplementedError

    def evaluate(self):
        repeats = self.run_settings.eval_settings.get("example_plot_repeats", 100)
        return_episode_rewards = self.run_settings.eval_settings.get("return_episode_rewards", True)
        seed_offset, logdir_root = self.get_seed_offset_and_logdir_root()
        self.io.set_iteration(0)
        log_dir = self.io.get_eval_dir(logdir_root)
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        self.env_params.training = False
        env = self.make_env(seed_offset, self.sim_setup, str(log_dir), env_params=self.env_params, training=False)()

        per_episode_rewards = []
        for n in range(repeats):
            done = False
            reward_sum = 0
            observation = env.reset()
            while not done:
                action = self.get_policy_action(observation)
                observation, reward, done, info = env.step(action)
                reward_sum += reward
            per_episode_rewards.append(reward_sum)
            self.reset()
        # Final reset to stash the last set of results
        env.reset()

        env.net.save_df(resolution=100)
        env.save_df()
        if return_episode_rewards:
            self.io.write_reward_file(per_episode_rewards, logdir_root)

    def get_action_space(self):
        temp_env = self.make_env(0, self.sim_setup, None, env_params=self.env_params, training=False)()
        return temp_env.action_space

    @staticmethod
    def get_valid_controller_settings():
        raise NotImplementedError

    # Not all simple controllers need reset.
    def reset(self):
        pass


class FixedControlTarget(NoLearningControlTarget):
    def __init__(self, trial_settings: CEPATrialSettings):
        super().__init__(trial_settings)
        self.action_length = self.get_action_space().shape[0]
        action_description = self.controller_settings.get("action", None)
        self.action = self.translate_action_description_to_action(action_description)

    def translate_action_description_to_action(self, description):
        raise NotImplementedError

    def get_policy_action(self, _):
        return self.action

    @staticmethod
    def get_valid_controller_settings():
        return ["action"]
