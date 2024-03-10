import argparse
import sys
import tqdm

from models.utils import plot_and_log_paths, plot_and_log_paths_3d

sys.path.append("models.utils")
from models import DT, PDT_oracle_plan
import numpy as np

def main(output_directory):
    env_name = "antmaze-medium-diverse-v2"
    ds = PDT_oracle_plan.SequenceManualPlanDataset(env_name)

    mean =ds.state_mean[0]
    std = ds.state_std[0]

    for idx in tqdm.tqdm(range(100), desc="Generating ds paths"):
        traj_idx = np.random.choice(len(ds.dataset), p=ds.sample_prob)
        trajectory = ds.dataset[traj_idx]
        observations = trajectory['observations']
        goal = DT.normalize_state(trajectory['goals'][0], mean[:2], std[:2])
        ant_path = DT.normalize_state(observations, mean, std)[:, :3]
        orientation_path = DT.normalize_state(observations, mean, std)[:, 3:6]
        start = ant_path[0]

        for i, plot_fn in enumerate([plot_and_log_paths, plot_and_log_paths_3d]):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot D4RL antmaze path data for visualization')
    parser.add_argument('output_directory', type=str, help='Directory to save plots')
    args = parser.parse_args()
    main(args.output_directory)