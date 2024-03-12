"""
Modified version of the single file implementation of Decision transformer as provided by the CORL team
"""

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy.*')

import os, sys

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ["WANDB_SILENT"] = "true"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Modify PDT forward pass to generate planning tokens which use embedded dim and thus are the same size and able to be
# concatenated

import d4rl  # noqa
from models.DT import *
from models.utils import *


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
    plan_length: int = 1
    num_plan_points: int = 10
    plan_bar_visualisation: bool = False
    replanning_interval: int = 40

    # video
    record_video: bool = False
    run_name: str = "run_0"
    bg_image: str = "./antmaze_medium_bg.png"
    num_eval_videos: int = 3

    #other
    checkpoint_to_load: Optional[str] = None
    checkpoint_step_to_load: Optional[int] = None
    eval_offline_every: int = 50
    eval_path_plot_every: int = 1000

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoint_to_load is not None:
            self.name = self.checkpoint_to_load
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class SequenceManualPlanDataset(SequenceDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0, path_length=10,
                 plan_length: int = 1, embedding_dim: int = 128):
        super().__init__(env_name, seq_len, reward_scale)
        self.path_length = path_length
        self.plan_length = plan_length
        self.embedding_dim = embedding_dim

        traj_dists = np.array([self.traj_distance(traj) for traj in self.dataset])
        # self.sample_prob = traj_dists / traj_dists.sum()
        self.sample_prob *= traj_dists / traj_dists.sum()
        self.sample_prob /= self.sample_prob.sum()
        self.expected_cum_reward = np.array([traj['returns'][0]*p for traj,p in zip(self.dataset,self.sample_prob)]).sum()
        self.max_seq_length = max(self.info["traj_lens"])

    @staticmethod
    def traj_distance(traj):
        obs = traj['observations']
        return np.linalg.norm(obs[-1][:2] - obs[0][:2])

    def create_plan(self, states):
        if self.plan_length:
            positions = states[:, :2]
            path = np.array(simplify_path_to_target_points_by_distance_log_scale(positions, self.path_length))[np.newaxis, :]
            path = pad_along_axis(path, pad_to=self.path_length, axis=1).reshape(-1, self.path_length * 2)
            return path
        else:
            return np.empty((0, self.path_length * 2))

    def _prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        goal = traj["goals"][0:1].astype(np.float32)

        # Hindsight goal relabelling only if the goal was actually achieved
        # Relabelling all trajectories can cause it to learn that bad actions still reach the goal
        # if traj["returns"][0:1]>0: goal = traj["observations"][-1:, :2]

        states = traj["observations"][start_idx: start_idx + self.seq_len]
        states_till_end = traj["observations"][start_idx:]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = normalize_state(states, self.state_mean, self.state_std)
        states_till_end = normalize_state(states_till_end, self.state_mean, self.state_std)
        goal = normalize_state(goal, self.state_mean[0:1, :2], self.state_std[0:1, :2])
        returns = returns * self.reward_scale
        plan = self.create_plan(states_till_end).astype(np.float32)
        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        steps_till_end = states_till_end.shape[0]
        if steps_till_end < self.max_seq_length:
            states_till_end = pad_along_axis(states_till_end, pad_to=self.max_seq_length)

        weight = (traj["returns"][0]+self.reward_scale)/(self.expected_cum_reward+self.reward_scale)

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
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len + plan_length + 1,  # Adjusted for the planning token and goal
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.plan_length = plan_length
        self.planning_head = nn.Sequential(nn.Linear(embedding_dim, plan_dim), nn.Tanh())
        self.plan_emb = nn.Linear(plan_dim, embedding_dim)
        self.plan_positional_emb = nn.Embedding(plan_length, embedding_dim)
        self.plan_dim = plan_dim
        # Create position IDs for plan once
        self.register_buffer('plan_position_ids', torch.arange(0, self.plan_length).unsqueeze(0))

        self.apply(self._init_weights)

    def forward(self, goal, states, actions, returns_to_go, time_steps, plan, padding_mask=None,
                log_attention=False):
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device

        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        plan_pos_emb = self.plan_positional_emb(self.plan_position_ids)
        state_emb_no_time_emb = self.state_emb(states)
        state_emb = state_emb_no_time_emb + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        plan_emb = self.plan_emb(plan) + plan_pos_emb if self.plan_length else \
            torch.empty(batch_size, 0, self.embedding_dim, device=device)

        # handle goal
        # we do this inserting the goal into the state, embedding it then subtracting the state embedding
        # we detatch the state embedding to prevent co-dependency during backprop
        goal_modified_state_0 = torch.cat((goal, states[:, 0:1, 2:].clone().detach()), dim=-1)
        goal_token = (self.state_emb(goal_modified_state_0) - state_emb_no_time_emb[:, 0:1, :]).detach()

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        # convert to form (goal, r_0, s_0, p_0, p1, ..., p_n, a_0, r_1, s_1, a_1, ...)
        sequence = construct_sequence_with_goal_and_plan(goal_token, plan_emb, sequence)

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
            # account for the planning token in the mask
            # True values in the mask mean don't attend to, so we use zeroes so the plan and goal are always attended to
            plan_mask = torch.zeros(batch_size, self.plan_length, dtype=torch.bool,
                                    device=device)
            goal_mask = torch.zeros(batch_size, 1, dtype=torch.bool,
                                    device=device)

            padding_mask = construct_sequence_with_goal_and_plan(goal_mask, plan_mask, padding_mask)
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        # for some interpretability lets get the attention maps
        attention_maps = []
        for i, block in enumerate(self.blocks):
            out, attn_weights = block(out, padding_mask=padding_mask, log_attention=log_attention)
            attention_maps.append(attn_weights)

        out = self.out_norm(out)

        # for input to the planning_head we use the sequence shiftted one to the left of the plan_sequence
        out_plan = self.planning_head(out[:, 2: 2 + self.plan_length])

        # predict actions only from state embeddings
        states = torch.cat([out[:, 2:3], out[:, (5 + self.plan_length)::3]], dim=1)
        out_actions = self.action_head(states) * self.max_action  # [batch_size, seq_len, action_dim]

        return out_plan, out_actions, attention_maps

# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
        pdt_model: PlanningDecisionTransformer,
        env: gym.Env,
        target_return: float,
        plan_length: int,
        device: str = "gpu",
        ep_id: int = 0,
        plan_bar_visualisation: bool = False,
        replanning_interval: int = 40,
        record_video: bool = False
) -> Tuple[float, float, list, list, list, list, np.ndarray, tuple]:
    states = torch.zeros(1, pdt_model.episode_len + 1, pdt_model.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, pdt_model.episode_len, pdt_model.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, pdt_model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(pdt_model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    print("START: ", states[0, 0][:3])
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0.0
    attention_map_frames = []
    render_frames = []
    pt_frames = []
    plan = None
    plan_paths = []
    goal_unmodified = env.target_goal
    print(goal_unmodified)
    goal = torch.tensor(goal_unmodified, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    for step in range(pdt_model.episode_len):
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
                plan[:, plan_token_i] = pred_plan[0, -1]

            if plan_length:
                plan_path = plan[0][0].detach().cpu()
                # Unflatten the last axis, so we have a 2D path
                plan_path = plan_path.reshape(*plan_path.shape[:-1], -1, 2)
                plan_paths.append(plan_path)
                if plan_bar_visualisation:
                    pt_frame = log_tensor_as_image(plan_path.view(-1), f"plan_ep_{ep_id}",
                                                   log_to_wandb=False)
                    pt_frames.append(pt_frame)

        if record_video:
            render_frames.append(env.render(mode="rgb_array"))
        _, predicted_actions, attention_maps = pdt_model(
            goal,
            states[:, : step + 1][:, -pdt_model.seq_len:],
            actions[:, : step + 1][:, -pdt_model.seq_len:],
            returns[:, : step + 1][:, -pdt_model.seq_len:],
            time_steps[:, : step + 1][:, -pdt_model.seq_len:],
            plan,
            log_attention=True
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)

        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if ep_id < 3:
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

    ant_path = states[0, :int(episode_len + 1), :2].cpu().numpy()
    return episode_return, episode_len, attention_map_frames, render_frames, pt_frames, plan_paths, ant_path, \
        goal_unmodified


@pyrallis.wrap()
def train(config: TrainConfig):
    num_cores = os.sysconf("SC_NPROCESSORS_ONLN")
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceManualPlanDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        path_length=config.num_plan_points,
        plan_length=config.plan_length,
        embedding_dim=config.embedding_dim
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=num_cores,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
        normalize_target=True
    )

    # DT model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    pdt_model = PlanningDecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        plan_dim=config.num_plan_points * 2,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
        plan_length=config.plan_length,
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
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    step_start = 0
    if config.checkpoint_to_load:
        step = config.checkpoint_step_to_load
        checkpoint = torch.load(os.path.join(config.checkpoints_path,f'pdt_checkpoint_step={step}.pt'))
        pdt_model.load_state_dict(checkpoint["pdt_model_state"])
        # dataset.state_mean, dataset.state_std = checkpoint["state_mean"], checkpoint["state_std"]
        # step_start = checkpoint["steps"] if "steps" in checkpoint else config.update_steps
        step_start = checkpoint["step"] if "step" in checkpoint else step

    print(f"Total parameters (DT): {sum(p.numel() for p in pdt_model.parameters())}")
    trainloader_iter = iter(trainloader)
    first_batch = None

    for step in trange(step_start, config.update_steps+step_start, desc="Training"):
        # print(f"step {step} in train loop")
        batch = next(trainloader_iter)
        if step == step_start:
            first_batch = batch
        if step % config.eval_offline_every == 0:
            batch = first_batch
        goal, states, actions, returns, time_steps, mask, plan, states_till_end, steps_left, return_weight = \
            [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

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
        # return_weight = return_weight[:, np.newaxis, np.newaxis]
        return_weight = 1.0

        plan_loss = F.mse_loss(predicted_plan, plan.detach(),reduction="none")
        plan_loss = (plan_loss * return_weight).mean()
        action_loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        action_loss = (action_loss * mask.unsqueeze(-1) * return_weight).mean()
        # action_loss = (action_loss * mask.unsqueeze(-1)).mean()
        combined_loss = 0.5 * action_loss + 0.5 * plan_loss if config.plan_length else action_loss
        # Backpropagation and optimization for main model

        optim.zero_grad()
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

        if step % config.eval_offline_every == 0:
            for batch_index in range(10):
                paths = []
                if config.plan_length:
                    # select batch_index of the batch and first of the plan sequence
                    for i, path in enumerate([predicted_plan, plan]):
                        transformed_path = path[batch_index, 0].detach().cpu()
                        paths.append(transformed_path.reshape(*transformed_path.shape[:-1], -1, 2))

                # for all variables we use the first of the batch
                plot_and_log_paths(
                    image_path=config.bg_image,
                    start=states[batch_index, 0, :2].cpu(),
                    goal=goal[batch_index, 0].cpu(),
                    plan_paths=paths,
                    ant_path=states_till_end[batch_index, :steps_left[batch_index], :2].cpu(),
                    output_folder=f'./visualisations/PDTv2-oracle-plan/{config.env_name}/{config.run_name}/train',
                    index=f"step={step}-batch_idx={batch_index}",
                    log_to_wandb=False,
                    pos_mean=dataset.state_mean[0, :2],
                    pos_std=dataset.state_std[0, :2]
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
        #  and step != 0
        if step == 0: continue
        if (step % config.eval_every == 0) or (
                step % config.eval_path_plot_every == 0) or step == config.update_steps - 1:
            if step % config.eval_path_plot_every == 0 and step % config.eval_every != 0:
                num_episodes = 1
            else:
                num_episodes = config.eval_episodes
            pdt_model.eval()
            for target_return in config.target_returns:
                if config.eval_seed != -1:
                    set_seed(config.eval_seed, eval_env, deterministic_torch=False)
                eval_returns = []
                eval_goal_dists = []
                for ep_id in trange(num_episodes, desc="Evaluation", leave=False):
                    if config.eval_seed == -1:
                        set_seed(range(config.eval_episodes)[ep_id] * 2, eval_env, deterministic_torch=False)
                    video_folder = f'./visualisations/PDTv2-oracle-plan/{config.env_name}/{config.run_name}'
                    os.makedirs(video_folder, exist_ok=True)
                    eval_return, eval_len, attention_frames, render_frames, plan_frames, plan_paths, ant_path, goal = eval_rollout(
                        pdt_model=pdt_model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        plan_length=config.plan_length,
                        device=config.device,
                        ep_id=ep_id,
                        plan_bar_visualisation=config.plan_bar_visualisation,
                        record_video=config.record_video and ep_id < config.num_eval_videos,
                        replanning_interval=config.replanning_interval
                    )
                    if len(attention_frames):
                        arrays_to_video(attention_frames, f'{video_folder}/attention_t={step}-ep={ep_id}.mp4',
                                    scale_factor=5)
                    if ep_id < config.num_eval_videos:
                        arrays_to_video(render_frames, f'{video_folder}/render_t={step}-ep={ep_id}.mp4', scale_factor=1,
                                    use_grid=False)
                    if config.plan_length and config.plan_bar_visualisation:
                        arrays_to_video(plan_frames, f'{video_folder}/planning-token_t={step}-ep={ep_id}.mp4',
                                        scale_factor=1, fps=1, use_grid=False)
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)
                    eval_goal_dists.append(np.linalg.norm(ant_path[-1] - goal))
                    plot_and_log_paths(
                        image_path=config.bg_image,
                        start=ant_path[0],
                        goal=goal,
                        plan_paths=plan_paths,
                        ant_path=ant_path,
                        output_folder=video_folder,
                        index=f"t={step}-ep={ep_id}",
                        log_to_wandb=False,
                        pos_mean=dataset.state_mean[0][:2],
                        pos_std=dataset.state_std[0][:2]
                    )

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                print(f"eval/{target_return}_return_mean", np.mean(eval_returns))
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
                        },
                        step=step,
                    )
            pdt_model.train()

            if config.checkpoints_path is not None:
                checkpoint = {
                    "pdt_model_state": pdt_model.state_dict(),
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "step": step
                }
                torch.save(checkpoint, os.path.join(config.checkpoints_path, f"pdt_checkpoint_step={step}.pt"))


if __name__ == "__main__":
    train()
