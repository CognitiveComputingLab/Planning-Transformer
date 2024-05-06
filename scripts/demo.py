from models.PDT import *
@pyrallis.wrap()
def demo(config: TrainConfig):
    # data & dataloader setup
    dataset = SequenceManualPlanDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        path_length=config.num_plan_points,
        embedding_dim=config.embedding_dim,
        use_log_distance=config.use_log_distance_plans,
        plan_type=config.plan_type,
        is_goal_conditioned=config.is_goal_conditioned,
        plans_use_actions=config.plans_use_actions,
        plan_indices=config.plan_indices
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
        use_goal_space_first_state_token=config.use_goal_space_first_state_token,
        use_timestep_embedding=config.use_timestep_embedding
    ).to(config.device)

    step = config.checkpoint_step_to_load
    checkpoint = torch.load(os.path.join(config.checkpoints_path, f'pdt_checkpoint_step={step}.pt'))
    pdt_model.load_state_dict(checkpoint["pdt_model_state"])
    # dataset.state_mean, dataset.state_std = checkpoint["state_mean"], checkpoint["state_std"]
    # step_start = checkpoint["steps"] if "steps" in checkpoint else config.update_steps
    step_start = checkpoint["step"] if "step" in checkpoint else step

    print(f"Total parameters (DT): {sum(p.numel() for p in pdt_model.parameters())}")

    set_seed(range(config.eval_episodes)[ep_id] * 2, eval_env, deterministic_torch=False)

    video_folder = f'./visualisations/PDTv2-oracle-plan/{config.env_name}/{config.run_name}'
    os.makedirs(video_folder, exist_ok=True)
    eval_return, eval_len, attention_frames, render_frames, plan_frames, plan_paths, ant_path, goal = eval_rollout(
        pdt_model=pdt_model,
        env=eval_env,
        target_return=target_return * config.reward_scale,
        plan_length=dataset.plan_length,
        device=config.device,
        ep_id=ep_id,
        plan_bar_visualisation=config.plan_bar_visualisation,
        record_video=config.record_video and ep_id < config.num_eval_videos,
        replanning_interval=config.replanning_interval,
        early_stop_step=config.eval_early_stop_step,
        action_noise_scale=config.action_noise_scale,
        state_noise_scale=config.state_noise_scale
    )
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
    print(f"Eval Return: ")
    eval_returns.append(eval_return / config.reward_scale)

    eval_goal_dists.append(
        np.linalg.norm(ant_path[-1, config.goal_indices] - np.array(goal)[list(config.goal_indices)])
    )
    ant_path_pos = ant_path[:, config.ant_path_viz_indices]
    plan_paths = [dataset.convert_plan_to_path(path, config.plan_path_viz_indices)
                  for path in plan_paths]
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