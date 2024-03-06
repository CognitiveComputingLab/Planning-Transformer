import argparse
from models.utils import plot_and_log_paths
import os
import re
import pickle
from tqdm import tqdm
def main(directory):

    path_data_files = [f for f in os.listdir(directory) if re.match(r'path_data_.+pkl', f)]
    for path_data_file in tqdm(path_data_files, "plotting paths"):
        # if not "15000-ep=9" in path_data_file:
        #     continue
        with open(os.path.join(directory, path_data_file), 'rb') as f:
            data = pickle.load(f)

        plot_and_log_paths(
            image_path="antmaze_medium_bg.png",
            start = data['start'],
            goal = data['goal'],
            plan_paths = data['plan_paths'],
            ant_path = data['ant_path'],
            output_folder = directory,
            index=path_data_file.split('path_data_')[1].split('.pkl')[0],
            log_to_wandb=False,
            save_data=False,
            pos_mean=data['mean'],
            pos_std=data['std']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot antmaze path data for visualisation')
    parser.add_argument('directory', type=str, help='Directory containing videos')
    args = parser.parse_args()
    main(args.directory)