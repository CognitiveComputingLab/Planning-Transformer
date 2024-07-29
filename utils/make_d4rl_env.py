import warnings
import numpy as np
import torch
from d4rl.offline_env import OfflineEnv, OfflineEnvWrapper
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from gym.utils import colorize

# Mapping of environments to their corresponding lerobot repo_id and score ranges
env_to_lerobot_repoid = {
    'gym_pusht/PushT-v0': ('lerobot/pusht', 1.0, 0.0),
    'aloha/AlohaInsertion-v0': ('lerobot/aloha_sim_insertion_human', 1.0, 0.0),
    'aloha/AlohaTransferCube-v0': ('lerobot/aloha_sim_transfer_cube_human', 1.0, 0.0),
    'xarm/XarmLift-v0': ('lerobot/xarm_lift_medium', 1.0, 0.0)
}


def create_lerobot_d4rl_wrapper(env, repo_id=None, ref_max_score=None, ref_min_score=None,
                                deprecated=False, deprecation_message=None, **kwargs):
    """
    Factory function to create a LerobotD4RLWrapper instance or return the original env if it's already an OfflineEnv.

    Args:
        env: The environment to be wrapped.
        repo_id: Repository ID pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """
    if isinstance(env, OfflineEnv):
        return env

    return LerobotD4RLWrapper(env, repo_id, ref_max_score, ref_min_score, deprecated, deprecation_message, **kwargs)


class LerobotD4RLWrapper(OfflineEnvWrapper):
    """
    Wrapper class for an environment to act as a D4RL offline environment using lerobot dataset.

    Args:
        env: The environment to be wrapped.
        repo_id: Repository ID pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """

    def __init__(self, env, repo_id=None, ref_max_score=None, ref_min_score=None,
                 deprecated=False, deprecation_message=None, **kwargs):
        super(LerobotD4RLWrapper, self).__init__(env)

        # Set defaults from the mapping dictionary if not provided
        if repo_id is None or ref_max_score is None or ref_min_score is None:
            if env.spec.id not in env_to_lerobot_repoid:
                raise ValueError(f"Environment {env.spec.id} not found in env_to_lerobot_repoid mapping.")
            default_repo_id, default_max, default_min = env_to_lerobot_repoid[env.spec.id]
            repo_id = repo_id or default_repo_id
            ref_max_score = ref_max_score or default_max
            ref_min_score = ref_min_score or default_min

        self.repo_id = repo_id
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score

        if deprecated:
            if deprecation_message is None:
                deprecation_message = "This environment is deprecated. Please use the most recent version of this environment."
            warnings.warn(colorize(deprecation_message, 'yellow'), stacklevel=2)

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_dataset(self):
        if self.repo_id is None:
            raise ValueError("Offline env not configured with a repository ID.")

        # Load lerobot dataset using the LeRobotDataset class
        dataset = LeRobotDataset(self.repo_id)

        data_dict = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timeouts': []
        }

        use_env_state  = 'observation.environment_state' in dataset[0]
        for data in dataset:  # Assuming the dataset can be iterated in this manner
            data_dict['observations'].append(
                torch.cat((data['observation.state'], data['observation.environment_state']))
                if use_env_state
                else data['observation.state']
            )
            data_dict['actions'].append(data['action'])
            data_dict['rewards'].append(
                data.get('reward', 0.0))  # Assuming reward is part of the dataset or default to 0
            data_dict['terminals'].append(data['next.done'])
            data_dict['timeouts'].append(False)

        data_dict = {k: np.array(v) for k, v in data_dict.items()}

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))

        return data_dict

    def render(self, mode="human", **kwargs):
        return self.env.render()

    def step(self, action):
        observation, reward, terminated, timeout, info = self.env.step(action)
        reward = reward if terminated or timeout else 0.0
        return observation, reward, terminated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]
