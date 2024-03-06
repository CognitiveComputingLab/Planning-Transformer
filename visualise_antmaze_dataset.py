import argparse
import sys

from models.utils import plot_and_log_paths

sys.path.append("models.utils")
from models import DT


def main(output_directory):
    env_name = "antmaze-medium-diverse-v2"
    traj, info = DT.load_d4rl_trajectories(env_name)
    mean = info["obs_mean"][0][:2]
    std = info["obs_std"][0][:2]

    for idx, trajectory in enumerate(traj[:10]):
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