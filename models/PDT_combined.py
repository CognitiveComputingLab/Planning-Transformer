"""
Modified version of the single file implementation of Decision transformer as provided by the CORL team
"""

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy.*')

import os

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ["WANDB_SILENT"] = "false"

# Modify PDT forward pass to generate planning tokens which use embedded dim and thus are the same size and able to be
# concatenated

import d4rl  # noqa
from DT import *
from utils import log_attention_maps, log_tensor_as_image, arrays_to_video


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

    # Add new fields for PTG
    ptg_num_layers: int = 3  # Number of transformer layers for PTG
    ptg_learning_rate: float = 1e-2  # Learning rate for PTG optimizer
    ptg_warmup_steps: int = warmup_steps # Warmup steps for PTG learning rate scheduler
    use_planning_token: bool = True
    num_planning_tokens: int = 5

    # video
    record_video: bool=False
    run_name: str = "run_0"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class PlanningDecisionTransformer(DecisionTransformer):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 seq_len: int = 10,
                 embedding_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 attention_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 use_planning_token: bool = True,
                 num_planning_tokens: int = 5,
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
                    seq_len=3 * seq_len + use_planning_token*num_planning_tokens,  # Adjusted for the planning token
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.use_planning_token = use_planning_token
        self.num_planning_tokens = num_planning_tokens

    def forward(self, states, actions, returns_to_go, time_steps, planning_token, padding_mask=None,
                log_attention=False):
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)

        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        if self.use_planning_token:
            sequence = torch.cat([planning_token, sequence], dim=1)

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
            if self.use_planning_token:
                # account for the planning token in the mask
                planning_token_mask = torch.zeros(batch_size, self.num_planning_tokens, dtype=torch.bool, device=planning_token.device)
                padding_mask = torch.cat([planning_token_mask, padding_mask], dim=1)

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
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, (1 + self.use_planning_token*self.num_planning_tokens)::3]) * self.max_action  # The "+ 1" accounts for the planning token
        return out, attention_maps


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
        pdt_model: PlanningDecisionTransformer,
        env: gym.Env,
        target_return: float,
        device: str = "gpu",
        use_planning_token: bool = True,
        ep_id = 0
) -> Tuple[float, float, list, list, list]:
    states = torch.zeros(
        1, pdt_model.episode_len + 1, pdt_model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, pdt_model.episode_len, pdt_model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, pdt_model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(pdt_model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len = 0.0, 0.0
    attention_map_frames = []
    render_frames = []
    pt_frames = []

    def model_call(step, is_planning_token, log_attention):
        return pdt_model(
            states[:, : step + 1][:, -pdt_model.seq_len:],
            actions[:, : step + 1][:, -pdt_model.seq_len:],
            returns[:, : step + 1][:, -pdt_model.seq_len:],
            time_steps[:, : step + 1][:, -pdt_model.seq_len:],
            None if is_planning_token else planning_token,
            is_planning_token=is_planning_token,
            log_attention=log_attention
        )

    for step in range(pdt_model.episode_len):
        planning_token = None
        if step % 20 == 0 and use_planning_token:
            # Generate the planning token every 20 steps (partial replanning)
            planning_token, _ = model_call(step, is_planning_token=1, log_attention=False)
            pt_frame = log_tensor_as_image(planning_token[0].view(-1), f"planning_token_ep_{ep_id}", log_to_wandb=False)
            pt_frames.append(pt_frame)

        if ep_id < 3:
            render_frames.append(env.render(mode="rgb_array"))

        predicted_actions, attention_maps = model_call(step, is_planning_token=0, log_attention=ep_id < 3)
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)

        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if ep_id < 3:
            attention_map_frames.append(log_attention_maps(attention_maps, log_to_wandb=False)[0])

        if done:
            break

    return episode_return, episode_len, attention_map_frames, render_frames, pt_frames

@pyrallis.wrap()
def train(config: TrainConfig):
    num_cores = os.sysconf("SC_NPROCESSORS_ONLN")

    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
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
    )
    # DT model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    pdt_model = PlanningDecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
        use_planning_token=config.use_planning_token,
        num_planning_tokens=config.num_planning_tokens
    ).to(config.device)

    optim = torch.optim.AdamW(
        pdt_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters (DT): {sum(p.numel() for p in pdt_model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        # print(f"step {step} in train loop")
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        planning_token = ptg_model(states[:, 0, :]) if config.use_planning_token else None

        # Forward pass through the model with planning token
        predicted_actions, attention_maps = pdt_model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            planning_token=planning_token,  # Include planning token in the model's forward pass
            padding_mask=padding_mask,
            log_attention=False
        )

        main_loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        main_loss = (main_loss * mask.unsqueeze(-1)).mean()

        # Backpropagation and optimization for main model

        optim.zero_grad()
        main_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(pdt_model.parameters(), config.clip_grad)
            if config.use_planning_token: torch.nn.utils.clip_grad_norm_(ptg_model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()
        #
        # # Log gradients
        # for name, param in pdt_model.named_parameters():
        #     if 'state_emb' in name and param.grad is not None:
        #         wandb.log({f"gradients_after/{name}": wandb.Histogram(param.grad.cpu().data.numpy())})

        # print("train_loss", main_loss.item())
        wandb.log(
            {
                "train_loss": main_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        #  and step != 0
        if (step % config.eval_every == 0) or step == config.update_steps - 1:
            pdt_model.eval()
            if config.use_planning_token: ptg_model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for ep_id in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    video_folder = f'./video/{"PDT" if config.use_planning_token else "DT"}_' \
                                            f'{config.env_name}/{config.run_name}/episode-{ep_id}'

                    eval_return, eval_len, attention_frames,render_frames, pt_frames = eval_rollout(
                        pdt_model=pdt_model,
                        ptg_model=ptg_model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                        use_planning_token=config.use_planning_token,
                        ep_id=ep_id
                    )
                    if ep_id < 3:
                        os.makedirs(video_folder, exist_ok=True)
                        arrays_to_video(attention_frames, f'{video_folder}/attention_t={step}.mp4', scale_factor=5)
                        arrays_to_video(render_frames, f'{video_folder}/render_t={step}.mp4', scale_factor=1)
                        if config.use_planning_token:
                            arrays_to_video(pt_frames, f'{video_folder}/planning-token_t={step}.mp4', scale_factor=1, fps=1)
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                print(f"eval/{target_return}_return_mean", np.mean(eval_returns))
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            pdt_model.train()
            if config.use_planning_token: ptg_model.train()

        if config.checkpoints_path is not None:
            checkpoint = {
                "pdt_model_state": pdt_model.state_dict(),
                "ptg_model_state": ptg_model.state_dict() if config.use_planning_token else None,
                "state_mean": dataset.state_mean,
                "state_std": dataset.state_std,
            }
            torch.save(checkpoint, os.path.join(config.checkpoints_path, "pdt_checkpoint.pt"))


if __name__ == "__main__":
    train()
