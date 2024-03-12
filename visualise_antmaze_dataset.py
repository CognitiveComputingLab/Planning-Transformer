import argparse
import sys
import tqdm
import torch

from models.utils import plot_and_log_paths, plot_and_log_paths_3d, log_tensor_as_image

sys.path.append("models.utils")
from models import DT, PDT_oracle_plan
import numpy as np
from PIL import Image
import os

def main(output_directory):
    env_name = "antmaze-medium-diverse-v2"
    ds = PDT_oracle_plan.SequenceManualPlanDataset(env_name)

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
        if len(observations) < 10: continue
        goal = traj['goals'][0]
        if traj["returns"][0:1] > 0: goal = observations[-1, :2]
        goal = DT.normalize_state(goal, mean[:2], std[:2])
        ant_path = DT.normalize_state(observations, mean, std)[:, :3]
        orientation_path = DT.normalize_state(observations, mean, std)[:, 3:6]
        start = ant_path[0]

        # print(goal, ant_path[-1, :2])
        # goal = ant_path[-1, :2]
        reached_goal = np.any(traj['returns'])
        returns.append(reached_goal)
        lengths.append(len(traj['returns']))

        # or np.linalg.norm(ant_path[-1, :2] - goal) < 0.5
        if reached_goal:
            # , plot_and_log_paths_3d
            for i, plot_fn in enumerate([plot_and_log_paths]):
                plot_fn(
                    image_path="antmaze_medium_bg.png",
                    start=start,
                    goal=goal,
                    plan_paths=[],  # Assuming no planned paths here, adjust as needed
                    ant_path=ant_path[:, :2] if i==0 else ant_path,
                    output_folder=output_directory,
                    index=str(idx),
                    log_to_wandb=False,
                    save_data=False,
                    pos_mean= mean[:2],
                    pos_std = std[:2],
                    orientation_path=orientation_path
                )
        # returns = log_tensor_as_image(torch.from_numpy(trajectory['returns']), log_to_wandb=False)

        # if reached_goal:
        #     print([int(x) for x in trajectory['returns'][-100:]])
        # Image.fromarray(returns).save(os.path.join(output_directory, f"debug_returns_{idx}.png"))
    # print(returns)
    print(len(returns))
    print(len(ds.dataset))
    print(np.sum(returns))
    print(np.sum(returns)/len(returns))
    print(np.sum(returns) / len(ds.dataset))
    print([x for x,y in zip(lengths,returns) if y])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot D4RL antmaze path data for visualization')
    parser.add_argument('output_directory', type=str, help='Directory to save plots')
    args = parser.parse_args()
    main(args.output_directory)