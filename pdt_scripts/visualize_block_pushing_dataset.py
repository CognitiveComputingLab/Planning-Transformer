import argparse
import sys
import tqdm
import os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from models import DT, PDT
from utils.plotting_funcs import plot_and_log_paths, plot_and_log_paths_3d
import numpy as np
from utils.make_d4rl_env import *
import gym
from envs.block_pushing import block_pushing_multimodal

def main(output_directory):
    env_name = "BlockPushMultimodal-v0"
    eval_env = BlockPushD4RLWrapper(
        gym.make("BlockPushMultimodal-v0", abs_action=False),
        data_path='data/block_pushing/block_pushing/multimodal_push_seed.zarr',
        ref_max_score=100.,
        ref_min_score=0.
    )
    train_data, val_data = DT.load_d4rl_trajectories(eval_env, gamma=1.0, test_size=0.02)
    ds = PDT.SequencePlanDataset(train_data["trajectories"], train_data["info"])

    # rewards_weighted = 0
    # for traj, p in zip(ds.dataset, ds.sample_prob):
    #     print(p, traj["returns"][0])
    #     rewards_weighted += p* traj["returns"][0]
    #
    # print((ds.sample_prob!=0).sum())
    # print(rewards_weighted)

    mean =ds.state_mean[0]
    std = ds.state_std[0]

    returns = []
    lengths = []
    for idx, traj in enumerate(tqdm.tqdm(ds.dataset, desc="Generating ds paths")):
        # traj_idx = np.random.choice(len(ds.dataset), p=ds.sample_prob)
        # trajectory = ds.dataset[traj_idx]
        observations = traj['observations']
        if len(observations) < 20: continue
        # ant_path = DT.normalize_state(observations, mean, std)[:, 8:10]
        ant_path = traj['actions']
        start = ant_path[0]

        # or np.linalg.norm(ant_path[-1, :2] - goal) < 0.5
        for i, plot_fn in enumerate([plot_and_log_paths, plot_and_log_paths_3d]):
            if i==1: continue
            plot_fn(
                image_path=None,
                start=start,
                goal=None,
                plan_paths=[],  # Assuming no planned paths here, adjust as needed
                ant_path=ant_path[:, :2] if i==0 else ant_path,
                output_folder=output_directory,
                index=str(idx),
                log_to_wandb=False,
                save_data=True,
                pos_mean= mean[:2],
                pos_std = std[:2],
                remove_axis=False
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot D4RL block_push data for visualization')
    parser.add_argument('output_dir', type=str, help='Directory to save plots')
    args = parser.parse_args()
    main(args.output_dir)