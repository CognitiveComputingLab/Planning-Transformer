import argparse
import sys
import tqdm

from models.utils import plot_and_log_paths

sys.path.append("models.utils")
from models import DT, PDT_oracle_plan
import numpy as np

def main(output_directory):
    env_name = "antmaze-medium-diverse-v2"
    dataset = PDT_oracle_plan.SequenceManualPlanDataset(env_name)

    mean =dataset.state_mean[0, :2]
    std = dataset.state_std[0, :2]

    for idx in tqdm.tqdm(range(100), desc="Generating dataset paths"):
        traj_idx = np.random.choice(len(dataset.dataset), p=dataset.sample_prob)
        trajectory = dataset.dataset[traj_idx]
        observations = trajectory['observations']
        goal = DT.normalize_state(trajectory['goals'][0], mean, std)
        ant_path = DT.normalize_state(observations[:, :2], mean, std)
        start = ant_path[0]

        plot_and_log_paths(
            image_path="antmaze_medium_bg.png",
            start=start,
            goal=goal,
            plan_paths=[],  # Assuming no planned paths here, adjust as needed
            ant_path=ant_path,
            output_folder=output_directory,
            index=str(idx),
            log_to_wandb=False,
            save_data=False,
            pos_mean= mean,
            pos_std = std
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot D4RL antmaze path data for visualization')
    parser.add_argument('output_directory', type=str, help='Directory to save plots')
    args = parser.parse_args()
    main(args.output_directory)