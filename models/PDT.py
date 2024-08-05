"""
Modified version of the single file implementation of Decision transformer as provided by the CORL team
"""

# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy.*')

import os, sys

# os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
# os.environ["WANDB_SILENT"] = "true"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import d4rl  # noqa
from models.DT import *
from utils.plotting_funcs import *
from utils.path_sampler import PathSampler
from math import inf
import re
from pdt_scripts.generate_demo_videos import generate_demo_video
from functools import partial
from d4rl.kitchen import kitchen_envs
from utils.make_d4rl_env import *
import gymnasium
import gym_pusht

from mujoco_py import GlfwContext

GlfwContext(offscreen=True)  # Create a window to init GLFW.

import mujoco_py

print(mujoco_py.cymj)


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "DT-D4RL"
    # wandb run name
    name: str = "PDT"
    # transformer hidden dim
    embedding_dim: int = 128
    # depth of the transformer model
    num_layers: int = 3
    # number of heads in the attention
    num_heads: int = 1
    # maximum sequence length during training
    seq_len: int = 20
    # maximum rollout length, needed for the positional embeddings
    episode_len: int = 1000
    # attention dropout
    attention_dropout: float = 0.1
    # residual dropout
    residual_dropout: float = 0.1
    # embeddings dropout
    embedding_dropout: float = 0.1
    # maximum range for the symmetric actions, [-1, 1]
    max_action: float = 1.0
    # training dataset and evaluation environment
    env_name: str = "halfcheetah-medium-v2"
    # AdamW optimizer learning rate
    learning_rate: float = 1e-4
    # AdamW optimizer betas
    betas: Tuple[float, float] = (0.9, 0.999)
    # AdamW weight decay
    weight_decay: float = 1e-4
    # maximum gradient norm during training, optional
    clip_grad: Optional[float] = 0.25
    # training batch size
    batch_size: int = 64
    # total training steps
    update_steps: int = 100_000
    # warmup steps for the learning rate scheduler
    warmup_steps: int = 10_000
    # reward scaling, to reduce the magnitude
    reward_scale: float = 0.001
    # number of workers for the pytorch dataloader
    num_workers: int = 4
    # target return-to-go for the prompting durint evaluation
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # number of episodes to run during evaluation
    eval_episodes: int = 100
    # evaluation frequency, will evaluate eval_every training steps
    eval_every: int = 10_000
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    train_seed: int = 10
    # evaluation random seed
    eval_seed: int = 42
    # training device
    device: str = "cuda"
    # log attention weights
    log_attn_weights: bool = False
    log_attn_every: int = 100

    # Options for planning tokens
    num_plan_points: int = 10
    plan_bar_visualisation: bool = False
    plan_use_full_state: Optional[bool] = False
    replanning_interval: int = 40
    plan_indices: Optional[Tuple[int, ...]] = None
    path_viz_indices: Optional[Tuple[int, ...]] = (0, 1)
    plans_use_actions: Optional[bool] = False
    non_plan_downweighting: Optional[float] = 0.0

    # video
    record_video: bool = False
    run_name: str = "run_0"
    bg_image: str = None
    num_eval_videos: int = 3

    # ablation testing parameters main
    plan_sampling_method: Optional[int] = 4  # 1: fixed-time", 2: fixed-distance", 3: log-time", 4: log-distance
    plan_use_relative_states: Optional[bool] = True  # States have first state subtracted
    goal_representation: Optional[
        int] = 3  # 1: Absolute Goal, 2: Relative goal, 3: State project to Goal and Absolute Goal
    plan_combine_observations: Optional[bool] = False  # Split observations into multiple tokens or combine
    plan_disabled: Optional[bool] = False  # turn off the plan

    # ablation testing parameters other
    plan_max_trajectory_ratio: Optional[float] = 0.5
    state_noise_scale: Optional[float] = 0.0
    use_two_phase_training: Optional[bool] = False
    is_goal_conditioned: Optional[bool] = False
    goal_indices: Optional[Tuple[int, ...]] = (0, 1)
    use_timestep_embedding: Optional[bool] = True

    # other
    checkpoint_to_load: Optional[str] = None
    checkpoint_step_to_load: Optional[int] = None
    eval_offline_every: int = 50
    eval_path_plot_every: int = 1000
    use_return_weighting: Optional[bool] = False
    eval_early_stop_step: Optional[int] = episode_len
    action_noise_scale: Optional[float] = 0.4
    repo_id: Optional[str] = None
    state_dim: Optional[int] = None
    disable_return_targets: Optional[bool] = False

    demo_mode: Optional[bool] = False

    def __post_init__(self):
        if self.demo_mode and self.checkpoint_to_load is None:
            regex = re.compile(r'^pdt_checkpoint_step=(\d+)\.pt$')
            latest_time = 0  # Initial value for comparison
            latest_folder_name = None

            for dirpath, _, filenames in os.walk(self.checkpoints_path):
                for filename in filenames:
                    if match := regex.match(filename):
                        # Get the modification time of the file
                        filepath = os.path.join(dirpath, filename)
                        mod_time = os.path.getmtime(filepath)
                        # Check if this file is more recent
                        if mod_time > latest_time:
                            latest_time = mod_time
                            latest_folder_name = dirpath.split('/')[-1]
                            self.checkpoint_step_to_load = int(match.group(1))
            self.checkpoint_to_load = latest_folder_name

        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoint_to_load is not None:
            self.name = self.checkpoint_to_load
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        if self.plan_indices is not None:
            c = [self.plan_indices.index(b_elem) for b_elem in self.path_viz_indices if b_elem in self.plan_indices]
            if len(c) == len(self.path_viz_indices):
                self.plan_path_viz_indices = c
            else:
                raise ValueError(f"path_viz_indices must be contained in plan_indices")
        else:
            self.plan_path_viz_indices = self.path_viz_indices
        self.ant_path_viz_indices = self.path_viz_indices

        if not self.is_goal_conditioned:
            self.goal_indices = []

        if self.demo_mode:
            self.run_name = 'demo'
            self.num_eval_videos = 3
            self.eval_early_stop_step = 500


class MakeGoalEnv(gym.Wrapper):
    def __init__(self, env, env_name, normalize, state_mean, state_std, goal_target=None):
        super().__init__(env)
        assert callable(normalize)
        self.env_name = env_name
        self.normalize = normalize
        self.state_mean = state_mean
        self.state_std = state_std
        self.goal_target = goal_target

    def get_target_goal(self, obs=None):
        # obs is already normalized, but goal isn't.
        # We must be careful to normalize goal without re-normalizing obs (which will break our goals)

        if self.goal_target is not None:
            return self.goal_target

        if "antmaze" in self.env_name:
            # print(env.target_goal)
            # return [-1.1, -1.1]
            # return [-1.8, -2.3]
            # return [0, 0]
            return normalize_state(self.env.target_goal, self.state_mean[0, :2], self.state_std[0, :2])
        if "kitchen" in self.env_name or "calvin" in self.env_name:
            goal = np.zeros(self.state_mean[0].shape) if obs is None else np.array(obs)

            if "kitchen" in self.env_name:
                for task in self.env.TASK_ELEMENTS:
                    indices = kitchen_envs.OBS_ELEMENT_INDICES[task]
                    values = kitchen_envs.OBS_ELEMENT_GOALS[task]
                    goal[indices] = normalize_state(values, self.state_mean[0][indices], self.state_std[0][indices])

            elif "calvin" in self.env_name:
                indices = slice(15, 21)
                values = np.array([0.25, 0.15, 0, 0.088, 1, 1])
                goal[indices] = normalize_state(values, self.state_mean[0][indices], self.state_std[0][indices])
            return goal

        return np.zeros((1, 1), dtype=np.float32)
        # raise ValueError("Expected antmaze or kitchen env, found ", env_id)


def obs_dict_to_vec(obs):
    if type(obs) is dict:
        # for handelling push-t's observation which returns a tuple
        return np.concatenate((obs['agent_pos'], obs['environment_state']))
    else:
        return obs


def wrap_goal_env(
        env: gym.Env,
        env_name: str,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
        goal_target: list = None,
) -> MakeGoalEnv:
    env = gym.wrappers.TransformObservation(env, partial(obs_dict_to_vec))
    env = gym.wrappers.TransformObservation(env, partial(normalize_state, state_mean=state_mean, state_std=state_std))
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, partial(scale_reward, reward_scale=reward_scale))
    env = MakeGoalEnv(env, env_name, normalize_state, state_mean, state_std, goal_target)
    return env


class SequencePlanDataset(SequenceDataset):
    def __init__(self, env: OfflineEnv, seq_len: int = 10, reward_scale: float = 1.0, path_length=10,
                 embedding_dim: int = 128, plan_sampling_method: int = 4, plan_max_trajectory_ratio=0.5,
                 plan_combine: bool = False, plan_disabled: bool = False, plan_indices: Tuple[int, ...] = (0, 1),
                 is_goal_conditioned: bool = False, plans_use_actions: bool = False):
        super().__init__(env, seq_len, reward_scale)
        self.path_length = path_length
        self.plan_indices = range(0, self.state_mean.shape[1]) if plan_indices is None else plan_indices
        self.plans_use_actions = plans_use_actions

        self.is_gc = is_goal_conditioned  # dataset depends on whether reward conditioned (rc) or goal conditioned (gc)
        self.plan_length = (1 if plan_combine else self.path_length) * (not plan_disabled)
        actions_shape = self.dataset[0]["actions"].shape[-1]
        self.path_dim = len(self.plan_indices) + (actions_shape if plans_use_actions else 0)
        self.plan_dim = (self.path_length if plan_combine else 1) \
                        * (not plan_disabled) \
                        * (self.path_dim + (not self.is_gc))
        self.embedding_dim = embedding_dim
        if self.is_gc:
            traj_dists = np.array([self.traj_distance(traj, plan_indices) for traj in self.dataset])
            # self.sample_prob = traj_dists / traj_dists.sum()
            self.sample_prob *= traj_dists / traj_dists.sum()
        self.sample_prob /= self.sample_prob.sum()

        self.expected_cum_reward = np.array(
            [traj['returns'][0] * p for traj, p in zip(self.dataset, self.sample_prob)]
        ).sum()
        self.max_traj_length = max(self.info["traj_lens"])
        self.plan_sampler = PathSampler(method=plan_sampling_method)
        self.plan_max_trajectory_ratio = plan_max_trajectory_ratio

        self.plan_combine = plan_combine

    @staticmethod
    def traj_distance(traj, indices):
        obs = traj['observations'][:, indices]
        return np.linalg.norm(obs[-1] - obs[0])

    def create_plan(self, states, returns=None, actions=None):
        if self.plan_length:
            positions = states[:, self.plan_indices]
            if returns is not None:
                # handle the reward conditioning
                positions = np.hstack((positions, returns[:, np.newaxis]))

            if actions is not None:
                positions = np.hstack((positions, actions))

            path = np.array(self.plan_sampler.sample(positions, self.path_length))
            if self.plan_combine:
                path = pad_along_axis(path[np.newaxis, :], pad_to=self.path_length, axis=1)
                path = path.reshape(-1, self.plan_dim)
            else:
                path = pad_along_axis(path, pad_to=self.plan_length, axis=0)
        else:
            path = np.empty((0, self.plan_dim))

        return path

    def convert_plan_to_path(self, plan, plan_path_viz_indices):
        if self.plan_combine:
            plan = plan[0].reshape(*plan[0].shape[:-1], -1, self.path_dim)

        return plan[:, plan_path_viz_indices]

    def _prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa

        # Hindsight goal relabelling only if the goal was actually achieved
        # Relabelling all trajectories can cause it to learn that bad actions still reach the goal
        # if traj["returns"][0:1]>0: goal = traj["observations"][-1:, :2]

        states = traj["observations"][start_idx: start_idx + self.seq_len]
        states_till_end = traj["observations"][start_idx:]

        # create the plan from the current state minus some random amount to at most half the max episode length
        # we subtract this random amount to make sure eval doesn't go OOD when the start state doesn't match.
        plan_states_start = max(0, start_idx + random.randint(-self.seq_len, 0))
        plan_states_end = start_idx + (
            max(math.floor(random.uniform(0.5, 1.0) * len(states_till_end)), self.path_length)
            if self.is_gc
            else int(self.max_traj_length * self.plan_max_trajectory_ratio)) + 1
        # plan_states_end = len(traj["observations"])
        plan_states = traj["observations"][plan_states_start:plan_states_end]
        plan_returns = traj["returns"][plan_states_start:plan_states_end] * self.reward_scale

        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len] * self.reward_scale
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = normalize_state(states, self.state_mean, self.state_std)
        states_till_end = normalize_state(states_till_end, self.state_mean, self.state_std)
        plan_states = normalize_state(plan_states, self.state_mean, self.state_std)
        if self.is_gc:
            # select random observation in future
            # since the plan already implements this logic we just select the last plan state
            goal = plan_states[-1:]
            # if "goals" in traj.keys():
            #     # ant maze specific fix in future
            #     goal = traj["goals"][0:1].astype(np.float32)
            #     goal = normalize_state(goal, self.state_mean[0:1, :2], self.state_std[0:1, :2])
            # else:
            #     # for other environments select random observation in future
            #     # since the plan already implements this logic we just select the last plan state
            #     goal = plan_states[-1:]
        else:
            goal = np.zeros((1, 1), dtype=np.float32)

        plan_states = plan_states[:int(self.max_traj_length * self.plan_max_trajectory_ratio)]
        plan_actions = traj["actions"][plan_states_start:plan_states_start + len(plan_states)] \
            if self.plans_use_actions else None
        if self.is_gc:
            plan = self.create_plan(plan_states, actions=plan_actions).astype(np.float32)
        else:
            plan_returns = plan_returns[:int(self.max_traj_length * self.plan_max_trajectory_ratio)]
            plan = self.create_plan(plan_states, returns=plan_returns, actions=plan_actions).astype(np.float32)

        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        steps_till_end = states_till_end.shape[0]
        if steps_till_end < self.max_traj_length:
            states_till_end = pad_along_axis(states_till_end, pad_to=self.max_traj_length)

        weight = (traj["returns"][0] + self.reward_scale) / (self.expected_cum_reward + self.reward_scale)

        return goal, states, actions, returns, time_steps, mask, plan, states_till_end, steps_till_end, weight


def construct_sequence_with_goal_and_plan(goal, plan, rsa_sequence):
    first_rs = rsa_sequence[:, :2]  # shape [batch_size, 2, emb_dim (or 1 if mask)]
    remaining_elements = rsa_sequence[:, 2:]  # shape [batch_size, 3*seq_len-2, emb_dim (or 1 if mask)]
    return torch.cat([goal, first_rs, plan, remaining_elements], dim=1)


def un_normalise_state(state, state_mean, state_std):
    return (state * state_std) + state_mean


class PlanningDecisionTransformer(DecisionTransformer):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 plan_dim: int,
                 seq_len: int = 10,
                 embedding_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 attention_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 plan_length: int = 1,
                 use_two_phase_training: bool = False,
                 goal_indices: Tuple[int, ...] = (0, 1),
                 plan_indices: Tuple[int, ...] = (0, 1),
                 non_plan_downweighting: float = 0.0,
                 use_timestep_embedding: bool = True,
                 plan_use_relative_states: bool = True,
                 goal_representation: int = 3,
                 **kwargs
                 ):
        super().__init__(state_dim, action_dim,
                         seq_len=seq_len,
                         embedding_dim=embedding_dim,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         attention_dropout=attention_dropout,
                         residual_dropout=residual_dropout,
                         **kwargs
                         )
        self.goal_cond = len(goal_indices) > 0
        self.goal_length = 0 if not self.goal_cond else 2 if goal_representation in [3, 4] else 1

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len + plan_length + self.goal_length,
                    # Adjusted for the planning token and goal
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.plan_length = plan_length
        self.non_plan_downweighting = non_plan_downweighting
        self.planning_head = nn.Linear(embedding_dim, plan_dim)
        self.plan_emb = nn.Linear(plan_dim, embedding_dim)
        self.goal_emb = nn.Linear(len(goal_indices), embedding_dim)
        self.plan_positional_emb = nn.Embedding(plan_length, embedding_dim)
        self.full_seq_pos_emb = nn.Embedding(3 + plan_length, embedding_dim)
        self.plan_dim = plan_dim
        self.use_two_phase_training = use_two_phase_training
        self.use_timestep = use_timestep_embedding
        self.goal_indices = torch.tensor(goal_indices, dtype=torch.long)
        self.plan_indices = torch.tensor(plan_indices, dtype=torch.long)

        # Create position IDs for plan once
        self.register_buffer('plan_position_ids', torch.arange(0, self.plan_length).unsqueeze(0))
        self.register_buffer('full_seq_pos_ids',
                             torch.arange(0,
                                          3 * seq_len + plan_length + self.goal_length).unsqueeze(
                                 0))

        # increase focus on plan by downweighting non plan tokens
        self.original_causal_masks = [block.causal_mask for block in self.blocks]

        self.apply(self._init_weights)

        # ablation testing variables for plan/goal representation
        self.plan_use_relative_states = plan_use_relative_states
        self.goal_representation = goal_representation

    def downweight_non_plan(self, plan_start, plan_length, downweighting):
        for i, block in enumerate(self.blocks):
            # attention takes a causal mask which is 0.0 if we fully attend and -inf to avoid attending
            # however by providing a value between -inf and 0.0 for the columns which are not the plans, we effectively
            # downweight the tokens attention towards non plan tokens, helping focus more on the plans
            new_attn_mask = torch.full(block.causal_mask.shape, downweighting, dtype=torch.float32)
            new_attn_mask[0:plan_start + plan_length, :] = 0.0
            new_attn_mask[:, plan_start:plan_start + plan_length] = 0.0
            new_attn_mask.masked_fill_(self.original_causal_masks[i], float('-inf')).fill_diagonal_(0.0)
            block.causal_mask = new_attn_mask.to(block.causal_mask.device)

    def forward(self, goal, states, actions, returns_to_go, time_steps, plan, padding_mask=None,
                log_attention=False):
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        # print(states.shape, self.state_emb.in_features, self.state_emb.out_features)
        state_emb_no_time_emb = self.state_emb(states)
        # state_emb_no_time_emb = self.plan_emb(states[:, :, :2])
        state_emb = state_emb_no_time_emb + (time_emb if self.use_timestep else 0)
        act_emb = self.action_emb(actions) + (time_emb if self.use_timestep else 0)
        # remove action conditioning (would this help?)
        # act_emb = torch.zeros (size=act_emb.shape, dtype=torch.float32, device=act_emb.device)
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + (time_emb if self.use_timestep else 0)
        for training_phase in range(self.use_two_phase_training + 1):
            plan_pos_emb = self.plan_positional_emb(self.plan_position_ids[:, :plan.shape[1]])
            # make plan relative, accounting for the possibility of actions in plan
            # can also add pos_emb here if don't want the full sequence embedding
            if self.plan_length:
                plan_states = plan[:, :, :len(self.plan_indices)].clone()
                if self.plan_use_relative_states:
                    plan_states -= states[:, :1, self.plan_indices]

                plan_emb = self.plan_emb(torch.cat((
                    plan_states,
                    plan[:, :, len(self.plan_indices):]
                ), dim=-1)) + plan_pos_emb

            else:
                plan_emb = torch.empty(batch_size, 0, self.embedding_dim, device=device)
            # plan_emb = self.plan_emb(plan) + plan_pos_emb if self.plan_length else \
            #     torch.empty(batch_size, 0, self.embedding_dim, device=device)

            # handle goal
            # we do this inserting the goal into the state, embedding it then subtracting the state embedding
            # we detatch the state embedding to prevent co-dependency during backprop
            # goal_modified_state_0 = states[:, 0:1].clone().detach()
            # goal_modified_state_0[:, :, self.goal_indices] = goal[:, :, self.goal_indices]
            # goal_token = (self.state_emb(goal_modified_state_0) - state_emb_no_time_emb[:, 0:1, :]).detach()
            # goal_token = torch.zeros(state_emb.shape, dtype=torch.float32, device=goal.device)[:,:1]
            # goal_token[:,:1,:2]=goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices]
            if self.goal_cond:
                if self.goal_representation == 1:
                    # absolute goal
                    goal_token = self.goal_emb(goal[:, :1, self.goal_indices])
                elif self.goal_representation == 2:
                    # relative goal
                    goal_token = self.goal_emb(goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices])
                elif self.goal_representation == 3:
                    # goal space state + absolute goal
                    goal_token = self.goal_emb(
                        torch.cat((states[:, :1, self.goal_indices],
                                   goal[:, :1, self.goal_indices]), dim=1))
                else:
                    # goal space state + relative goal
                    goal_token = self.goal_emb(
                        torch.cat((states[:, :1, self.goal_indices],
                                   goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices]), dim=1))
            else:
                goal_token = torch.empty(batch_size, 0, self.embedding_dim, device=device)

            # if(batch_size == 1):
            #     print(goal_token)
            # goal_token = self.goal_emb(goal[:, :1, self.goal_indices])

            # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
            sequence = (
                torch.stack([returns_emb, state_emb, act_emb], dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 3 * seq_len, self.embedding_dim)
            )
            # convert to form (goal, r_0, s_0, p_0, p1, ..., p_n, a_0, r_1, s_1, a_1, ...)
            sequence = construct_sequence_with_goal_and_plan(goal_token, plan_emb, sequence)
            # sequence[:, :2 + plan.shape[1]] += self.full_seq_pos_emb(self.full_seq_pos_ids[:, :2 + plan.shape[1]])
            padding_mask_full = None
            if padding_mask is not None:
                # [batch_size, seq_len * 3], stack mask identically to fit the sequence
                padding_mask_full = (
                    torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                    .permute(0, 2, 1)
                    .reshape(batch_size, 3 * seq_len)
                )
                # account for the planning token in the mask
                # True values in the mask mean don't attend to, so we use zeroes so the plan and goal are always attended to
                plan_mask = torch.zeros(plan_emb.shape[:2], dtype=torch.bool,
                                        device=device)
                goal_mask = torch.zeros(goal_token.shape[:2], dtype=torch.bool,
                                        device=device)

                padding_mask_full = construct_sequence_with_goal_and_plan(goal_mask, plan_mask, padding_mask_full)

            # LayerNorm and Dropout (!!!) as in original implementation,
            # while minGPT & huggingface uses only embedding dropout
            out = self.emb_norm(sequence)
            out = self.emb_drop(out)

            # for some interpretability lets get the attention maps
            attention_maps = []
            for i, block in enumerate(self.blocks):
                out, attn_weights = block(out, padding_mask=padding_mask_full, log_attention=log_attention)
                attention_maps.append(attn_weights)

            out = self.out_norm(out)

            start = 1 + self.goal_length
            if training_phase == 0:
                # for input to the planning_head we use the sequence shifted one to the left of the plan_sequence
                plan = self.planning_head(out[:, start: start + self.plan_length])
            if training_phase == self.use_two_phase_training:
                # predict actions only from the state embeddings
                out_states = torch.cat([out[:, start:start + 1], out[:, (start + 3 + self.plan_length)::3]], dim=1)
                out_actions = self.action_head(out_states) * self.max_action  # [batch_size, seq_len, action_dim]
        return plan, out_actions, attention_maps


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
        pdt_model: PlanningDecisionTransformer,
        env: MakeGoalEnv,
        target_return: float,
        plan_length: int,
        device: str = "gpu",
        ep_id: int = 3,
        plan_bar_visualisation: bool = False,
        replanning_interval: int = 40,
        record_video: bool = False,
        early_stop_step: int = inf,
        action_noise_scale: float = 0.7,
        state_noise_scale: float = 0.1,
        num_eps_with_logged_attention: int = 0,
        is_goal_conditioned: bool = False,
        disable_return_targets: bool = False
) -> Tuple[float, float, list, list, list, list, np.ndarray, tuple, float]:
    states = torch.zeros(1, pdt_model.episode_len + 1, pdt_model.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, pdt_model.episode_len, pdt_model.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, pdt_model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(pdt_model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    # print("START: ", states[0, 0][:3])
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0.0
    attention_map_frames = []
    attention_map_all_raw = []
    render_frames = []
    pt_frames = []
    plan = None
    plan_paths = []
    goal_unmodified = env.get_target_goal(obs=states[0, 0].cpu())
    goal = torch.tensor(goal_unmodified, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    alt_return = {
        "max_return": -inf,
        "final_return": 0
    }


    if pdt_model.non_plan_downweighting < 0:
        pdt_model.downweight_non_plan(2 + pdt_model.goal_length, pdt_model.plan_length,
                                      pdt_model.non_plan_downweighting)

    for step in range(min(pdt_model.episode_len, early_stop_step)):
        # Generate the planning token every 20 steps (partial replanning)
        if step % replanning_interval == 0:
            plan = torch.zeros(1, plan_length, pdt_model.plan_dim, dtype=torch.float, device=device)
            # plan tokens are generated via an autoregressive generation loop just like is done in NLP generation tasks
            for plan_token_i in range(plan_length):
                pred_plan, _, _ = pdt_model(
                    goal,
                    states[:, step:step + 1],
                    torch.zeros(1, 1, pdt_model.action_dim, dtype=torch.float, device=device),
                    returns[:, step:step + 1],
                    time_steps[:, step:step + 1],
                    plan=plan[:, :plan_token_i],
                    log_attention=False
                )
                plan[:, plan_token_i] = pred_plan[0, plan_token_i]
                # if plan_token_i == 1:
                #     plan[:, plan_token_i] = 0.2*goal+0.8*states[:, step:step + 1][:, :, :2]

            # if step==0:
            #     true_plan_2 = torch.stack(
            #         [(x*goal[0,0]+(1-x)*states[0, step, :2]) for x in torch.linspace(0.0,1.0,10)],
            #         dim=0
            #     ).unsqueeze(0)
            #     pred_plan_2, _, _ = pdt_model(
            #         goal,
            #         states[:, step:step + 1],
            #         torch.zeros(1, 1, pdt_model.action_dim, dtype=torch.float, device=device),
            #         torch.tensor([[0.5]], dtype=torch.float, device=device),
            #         time_steps[:, 900:901],
            #         plan=true_plan_2,
            #         log_attention=False
            #     )
            #     print(true_plan_2, pred_plan_2)

            if plan_length:
                plan_path = plan[0].detach().cpu()
                # Unflatten the last axis, so we have a 2D path
                plan_paths.append(plan_path)
                if plan_bar_visualisation:
                    pt_frame = log_tensor_as_image(plan_path.view(-1), f"plan_ep_{ep_id}",
                                                   log_to_wandb=False)
                    pt_frames.append(pt_frame)
        if record_video:
            frame = env.render(mode="rgb_array")
            render_frames.append(frame)
            # print(step)

        # actions_noisy = actions + torch.randn(actions.shape, device=device) * action_noise_scale * 0.5
        # states_noisy = torch.zeros(size=states.shape,dtype=torch.float32, device=states.device )
        # states_noisy = states_noisy[:, : step + 1][:, -pdt_model.seq_len:]
        # states_noisy[0] = states[:, : step + 1][:, -pdt_model.seq_len:][0]
        _, predicted_actions, attention_maps = pdt_model(
            goal,
            states[:, : step + 1][:, -pdt_model.seq_len:],
            # states_noisy,
            # actions_noisy[:, : step + 1][:, -pdt_model.seq_len:],
            actions[:, : step + 1][:, -pdt_model.seq_len:],
            returns[:, : step + 1][:, -pdt_model.seq_len:],
            time_steps[:, : step + 1][:, -pdt_model.seq_len:],
            plan,
            log_attention=True
        )
        attention_map_all_raw.append(np.array([x.cpu() for x in attention_maps]))
        predicted_action = predicted_actions[0, -1].cpu().numpy()

        if step < pdt_model.seq_len + 1:
            noise = 0.0
        else:
            attention_map_dist = np.linalg.norm(attention_map_all_raw[-1] - attention_map_all_raw[-2])
            noise = min(1.0, max(action_noise_scale, (attention_map_dist - 3.5) / (2.5 - 3.5)))

        if action_noise_scale != 0:
            # action_noise = np.random.normal(size=predicted_action.shape, scale=action_noise_scale)
            action_noise = np.random.normal(size=predicted_action.shape, scale=noise)
        else:
            action_noise = 0

        next_state, reward, done, info = env.step(predicted_action + action_noise)
        # next_state, reward, done, info = env.step(predicted_action)

        actions[:, step] = torch.as_tensor(predicted_action + action_noise)
        # actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - (0 if disable_return_targets else reward))

        episode_return += reward
        alt_return["max_return"] = max(alt_return["max_return"], reward)
        alt_return["final_return"] = reward
        episode_len += 1

        if ep_id < num_eps_with_logged_attention:
            attention_map_frames.append(log_attention_maps(attention_maps, log_to_wandb=False))

        if done:
            break

        # recent_states = states[0,step-29:step+1,:2]
        # dist = torch.sum(torch.norm(torch.diff(recent_states, dim=0), dim=1))
        # if step>100 and dist < 0.0005*30:
        #     print(f"agent frozen. Distance last 25 states = {dist}")
        #     # Pickle the numpy array
        #     with open('debug_data.pickle', 'wb') as f:
        #         pickle.dump({'states': states, 'actions': actions, 'plan': plan, 'returns': returns, 'step': step,
        #                      'ep_id':ep_id}, f)
        #     for i, img in enumerate(log_attention_maps(attention_maps, log_to_wandb=False)):
        #         img = Image.fromarray(img).resize(tuple(np.array(img.shape[:2])*5), resample=Image.NEAREST)
        #         img.save(f'debug_attention_map_{i}.png')
        #
        #     Image.fromarray(env.render(mode="rgb_array")).save("debug_render.png")
        #
        #     break

    # # debug attention maps for selective noise testing (REMOVE after)
    # with open(f"./visualisations/debug_attention/attention_maps_{ep_id}.pkl", 'wb') as f:
    #     pickle.dump(attention_map_all_raw, f)

    if pdt_model.non_plan_downweighting < 0:
        pdt_model.downweight_non_plan(2 + pdt_model.goal_length, pdt_model.plan_length, 0.0)

    ant_path = states[0, :int(episode_len + 1)].cpu().numpy()
    return episode_return, episode_len, attention_map_frames, render_frames, pt_frames, plan_paths, ant_path, \
        goal_unmodified, alt_return


@pyrallis.wrap()
def train(config: TrainConfig):
    num_cores = os.sysconf("SC_NPROCESSORS_ONLN")
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # environment setup uses code from https://github.com/seohongpark/HIQL/blob/master/main.py
    if 'antmaze' in config.env_name:
        if 'ultra' in config.env_name:
            sys.path.append('envs')
            import d4rl_ext
        eval_env = gym.make(config.env_name)
        eval_env.render(mode='rgb_array', width=200, height=200)

        dist, lookat = (70, (26, 18)) if 'ultra' in config.env_name else (50, (18, 12))
        eval_env.viewer.cam.lookat[:2] = lookat
        eval_env.viewer.cam.distance = dist
        eval_env.viewer.cam.elevation = -90
    elif 'kitchen' in config.env_name:
        eval_env = gym.make(config.env_name)
    elif config.env_name in lerobot_envs:
        eval_env = LerobotD4RLWrapper(
            gymnasium.make(
                config.env_name,
                obs_type="environment_state_agent_pos" if "pusht" in config.env_name else "state",
                render_mode="rgb_array",
                visualization_height=384,
                visualization_width=384,
            ),
            repo_id=config.repo_id
        )
    elif 'calvin' in config.env_name:
        from envs.calvin import CalvinEnv
        from hydra import compose, initialize

        # Initialize and configure the Calvin environment
        initialize(config_path='../envs/conf')
        cfg = compose(config_name='calvin')
        eval_env = CalvinEnv(**cfg)
        eval_env.max_episode_steps = cfg.max_episode_steps = 360

        eval_env = CalvinD4RLWrapper(
            eval_env,
            data_path='data/calvin.gz',  # Assuming config has a data_path attribute
            ref_max_score=4.,  # Assuming config has these attributes
            ref_min_score=0.
        )
    elif 'BlockPush' in config.env_name:
        from envs.block_pushing import block_pushing_multimodal

    else:
        eval_env = gym.make(config.env_name)

    # data & dataloader setup
    dataset = SequencePlanDataset(
        eval_env,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        path_length=config.num_plan_points,
        embedding_dim=config.embedding_dim,
        plan_sampling_method=config.plan_sampling_method,
        plan_combine=config.plan_combine_observations,
        plan_disabled=config.plan_disabled,
        is_goal_conditioned=config.is_goal_conditioned,
        plans_use_actions=config.plans_use_actions,
        plan_indices=config.plan_indices,
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=num_cores,
        timeout=10  # Set a timeout to catch hanging issues
    )
    goal_target = None
    if config.demo_mode:
        goal_target = np.array(demo_goal_select(
            image_path=config.bg_image,
            pos_mean=dataset.state_mean[0, config.ant_path_viz_indices],
            pos_std=dataset.state_std[0, config.ant_path_viz_indices],
        ))
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_goal_env(
        env=eval_env,
        env_name=config.env_name,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
        goal_target=goal_target
    )

    # DT model & optimizer & scheduler setup
    config.state_dim = config.state_dim or eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    pdt_model = PlanningDecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        plan_dim=dataset.plan_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
        plan_length=dataset.plan_length,
        use_two_phase_training=config.use_two_phase_training,
        goal_indices=config.goal_indices,
        plan_indices=range(0, dataset.state_mean.shape[1]) if config.plan_indices is None else config.plan_indices,
        non_plan_downweighting=config.non_plan_downweighting,
        use_timestep_embedding=config.use_timestep_embedding,
        plan_use_relative_states=config.plan_use_relative_states,
        goal_representation=config.goal_representation
    ).to(config.device)

    optim = torch.optim.AdamW(
        pdt_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: 1 if config.checkpoint_to_load else min((steps + 1) / config.warmup_steps, 1),
    )

    # save config to the checkpoint
    if config.checkpoints_path is not None and not config.demo_mode:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    step_start = 0
    if config.checkpoint_to_load:
        step = config.checkpoint_step_to_load
        checkpoint = torch.load(os.path.join(config.checkpoints_path, f'pdt_checkpoint_step={step}.pt'))
        pdt_model.load_state_dict(checkpoint["pdt_model_state"])
        # dataset.state_mean, dataset.state_std = checkpoint["state_mean"], checkpoint["state_std"]
        # step_start = checkpoint["steps"] if "steps" in checkpoint else config.update_steps
        step_start = checkpoint["step"] if "step" in checkpoint else step

    print(f"Total parameters (DT): {sum(p.numel() for p in pdt_model.parameters())}")
    trainloader_iter = iter(trainloader)
    # first_batch = None

    for step in trange(step_start, config.update_steps + step_start, desc="Training"):
        # print(f"step {step} in train loop")
        batch = next(trainloader_iter)

        # if step == step_start:
        #     first_batch = batch
        # if step % config.eval_offline_every == 0:
        #     batch = first_batch
        goal, states, actions, returns, time_steps, mask, plan, states_till_end, steps_left, return_weight = \
            [b.to(config.device) for b in batch]

        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        # increase focus on plan by corrupting states/actions during first 10k steps
        # if step / config.update_steps < 0.1:
        #     states[:, 1:] = torch.zeros(size=states.shape, dtype=torch.float32, device=states.device)[:, 1:]
        # actions_input = torch.zeros(size=actions.shape, dtype=torch.float32,
        #                             device=states.device) if step / config.update_steps < 0.1 else actions

        # every other gradient update step, corrupt states with empty 0s, to help focus on plans
        # if step % 2== 0:
        #     states[:, 1:] = torch.zeros(size=states.shape, dtype=torch.float32, device=states.device)[:, 1:]

        # Forward pass through the model with planning tokens
        predicted_plan, predicted_actions, attention_maps = pdt_model(
            goal=goal,
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            plan=plan,  # Include planning tokens in the model's forward pass
            padding_mask=padding_mask,
            log_attention=False,
        )

        # simple advantage weighting to encourage high return behaviour to be learnt
        return_weight = return_weight[:, np.newaxis, np.newaxis] if config.use_return_weighting else 1.0

        plan_loss = F.mse_loss(predicted_plan, plan.detach(), reduction="none")
        plan_loss = (plan_loss * return_weight).mean()
        action_loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        action_loss = (action_loss * mask.unsqueeze(-1) * return_weight).mean()
        combined_loss = 0.5 * action_loss + 0.5 * plan_loss if dataset.plan_length else action_loss
        # t = step / config.update_steps
        # combined_loss = t * action_loss + (1 - t) * plan_loss if dataset.plan_length else action_loss
        # if step/config.update_steps <0.1:
        #     combined_loss = plan_loss if dataset.plan_length else action_loss
        # else:
        #     combined_loss = 0.5 * action_loss + 0.5 * plan_loss if dataset.plan_length else action_loss
        # Backpropagation and optimization for main model

        optim.zero_grad()
        if not config.demo_mode:
            if not step % config.eval_offline_every == 0:
                combined_loss.backward()

            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(pdt_model.parameters(), config.clip_grad)
            optim.step()
            scheduler.step()
        #
        # # Log gradients
        # for name, param in pdt_model.named_parameters():
        #     if 'state_emb' in name and param.grad is not None:
        #         wandb.log({f"gradients_after/{name}": wandb.Histogram(param.grad.cpu().data.numpy())})

        # print("train_loss", main_loss.item())

        if step % config.eval_offline_every == 0 and not config.demo_mode:
            if len(config.ant_path_viz_indices) >= 2:
                for batch_index in random.sample(range(config.batch_size), 10):
                    paths = []
                    if dataset.plan_length:
                        # select batch_index of the batch and first of the plan sequence
                        for i, plan_instance in enumerate([predicted_plan, plan]):
                            plan_instance = plan_instance[batch_index].detach().cpu()
                            paths.append(dataset.convert_plan_to_path(plan_instance, config.plan_path_viz_indices))

                    # for all variables we use the first of the batch
                    plot_and_log_paths(
                        image_path=config.bg_image,
                        start=states[batch_index, 0, config.ant_path_viz_indices].cpu(),
                        goal=goal[batch_index, 0].cpu()[
                            np.array(config.path_viz_indices)] if config.is_goal_conditioned else None,
                        plan_paths=paths,
                        ant_path=states_till_end[batch_index, :steps_left[batch_index],
                                 config.ant_path_viz_indices].cpu(),
                        output_folder=f'./visualisations/PDTv2-oracle-plan/{config.env_name}/{config.run_name}/train',
                        index=f"step={step}-batch_idx={batch_index}",
                        log_to_wandb=False,
                        pos_mean=dataset.state_mean[0, config.ant_path_viz_indices],
                        pos_std=dataset.state_std[0, config.ant_path_viz_indices]
                    )

        wandb.log(
            {
                "train_action_loss": action_loss.item(),
                "train_plan_loss": plan_loss.item(),
                "train_combined_loss": combined_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step == 0 and not config.demo_mode: continue
        if config.demo_mode or (step % config.eval_every == 0) or (
                step % config.eval_path_plot_every == 0) or step == config.update_steps - 1:
            if step % config.eval_path_plot_every == 0 and step % config.eval_every != 0:
                num_episodes = 1
            else:
                num_episodes = config.eval_episodes

            if config.demo_mode: num_episodes = 3

            pdt_model.eval()
            video_folder = f'./visualisations/PDTv2-oracle-plan/{config.env_name}/{config.run_name}'
            for target_return in config.target_returns:
                # if config.eval_seed != -1:
                #     set_seed(config.eval_seed, eval_env, deterministic_torch=False)
                eval_returns = []
                eval_final_rewards = []
                eval_goal_dists = []
                for ep_id in trange(num_episodes, desc="Evaluation", leave=False):
                    if config.eval_seed == -1:
                        set_seed(range(config.eval_episodes)[ep_id] * 2, eval_env, deterministic_torch=False)

                    os.makedirs(video_folder, exist_ok=True)
                    eval_return, eval_len, attention_frames, render_frames, plan_frames, plan_paths, ant_path, goal, alt_returns = eval_rollout(
                        pdt_model=pdt_model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        plan_length=dataset.plan_length,
                        device=config.device,
                        ep_id=ep_id,
                        plan_bar_visualisation=config.plan_bar_visualisation,
                        record_video=config.record_video and ep_id < config.num_eval_videos and step % config.eval_every == 0,
                        replanning_interval=config.replanning_interval,
                        early_stop_step=config.eval_early_stop_step,
                        action_noise_scale=config.action_noise_scale,
                        state_noise_scale=config.state_noise_scale,
                        disable_return_targets=config.disable_return_targets
                    )
                    # use max_reward IoU
                    if 'pusht' in config.env_name:
                        eval_return = alt_returns["max_return"]

                    if len(attention_frames):
                        arrays_to_video(attention_frames, f'{video_folder}/attention_t={step}-ep={ep_id}.mp4',
                                        scale_factor=5)
                    if ep_id < config.num_eval_videos:
                        arrays_to_video(render_frames, f'{video_folder}/render_t={step}-ep={ep_id}.mp4', scale_factor=1,
                                        use_grid=False)
                    if dataset.plan_length and config.plan_bar_visualisation:
                        arrays_to_video(plan_frames, f'{video_folder}/planning-token_t={step}-ep={ep_id}.mp4',
                                        scale_factor=1, fps=1, use_grid=False)
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)
                    eval_final_rewards.append(alt_returns["final_return"] / config.reward_scale)

                    eval_goal_dists.append(
                        np.linalg.norm(ant_path[-1, config.goal_indices] - np.array(goal)[list(config.goal_indices)])
                    )
                    if len(config.ant_path_viz_indices) >= 2:
                        ant_path_pos = ant_path[:, config.ant_path_viz_indices]
                        plan_paths = [dataset.convert_plan_to_path(path, config.plan_path_viz_indices)
                                      for path in plan_paths]
                        # print(goal, np.array(config.path_viz_indices))
                        plot_and_log_paths(
                            image_path=config.bg_image,
                            start=ant_path_pos[0],
                            goal=goal[np.array(config.path_viz_indices)] if config.is_goal_conditioned else None,
                            plan_paths=plan_paths,
                            ant_path=ant_path_pos,
                            output_folder=video_folder,
                            index=f"t={step}-ep={ep_id}",
                            log_to_wandb=False,
                            pos_mean=dataset.state_mean[0, config.ant_path_viz_indices],
                            pos_std=dataset.state_std[0, config.ant_path_viz_indices],
                            orientation_path=ant_path[:, 3:7] if "antmaze" in config.env_name else None
                        )

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                print(f"eval/{target_return}_return_mean", np.mean(eval_returns))
                print(f"eval/{target_return}_normalized_score_mean", np.mean(normalized_scores))
                print(f"eval/{target_return}_final_reward_mean", np.mean(eval_final_rewards))
                if num_episodes == config.eval_episodes:
                    wandb.log(
                        {
                            f"eval/{target_return}_return_mean": np.mean(eval_returns),
                            # f"eval/{target_return}_return_std": np.std(eval_returns),
                            f"eval/{target_return}_normalized_score_mean": np.mean(
                                normalized_scores
                            ),
                            f"eval/{target_return}_normalized_score_std": np.std(
                                normalized_scores
                            ),
                            f"eval/{target_return}_goal_dist_mean": np.mean(eval_goal_dists),
                            f"eval/{target_return}_goal_dist_std": np.std(eval_goal_dists),
                            **({f"eval/{target_return}_final_reward_mean": np.mean(
                                eval_final_rewards)} if 'pusht' in config.env_name else {})
                        },
                        step=step,
                    )

            pdt_model.train()

            if config.demo_mode:
                generate_demo_video(video_folder)
                print("demo complete")
                return

            if config.checkpoints_path is not None and not config.demo_mode:
                checkpoint = {
                    "pdt_model_state": pdt_model.state_dict(),
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "step": step
                }
                torch.save(checkpoint, os.path.join(config.checkpoints_path, f"pdt_checkpoint_step={step}.pt"))


if __name__ == "__main__":
    train()
