import argparse
import cv2
import os
import pickle
from pdt_scripts import overlay_attention
from tqdm import tqdm
from models.utils import plot_and_log_paths
from PIL import Image
import re
import matplotlib
import numpy as np

def stack_images(img1, img2, output_path=None):
    target_width = max(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (target_width, int(img1.shape[0] * target_width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (target_width, int(img2.shape[0] * target_width / img2.shape[1])))
    new_img = np.vstack((img1_resized, img2_resized))
    if output_path is not None:
        cv2.imwrite(output_path, new_img)
    else:
        return new_img
def generate_demo_video(video_folder):
    overlay_attention.main(video_folder, 0.5)
    path_data_files = [f for f in os.listdir(video_folder) if re.match(r'path_data_.+pkl', f)][:3]

    for path_data_file in path_data_files:
        with open(os.path.join(video_folder, path_data_file), 'rb') as f:
            data = pickle.load(f)
        suffix = path_data_file.split('path_data_')[1].split('.pkl')[0]
        video_path = os.path.join(video_folder, f'combined_{suffix}.mp4')
        cap = cv2.VideoCapture(video_path)

        replanning_interval = 1 if not data['plan_paths'] else len(data['ant_path'])//len(data['plan_paths'])
        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_output_path = os.path.join(video_folder, f'demo_video_{suffix}.mp4')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = None
        plot_frame = None
        for frame_idx in tqdm(range(frame_count), 'generating video'):
            ret, render_frame = cap.read()
            if not ret:
                break  # Exit the loop if no frame is returned

            if frame_idx %3==0:
                plot_frame = plot_and_log_paths(
                    image_path="bg_images/antmaze_medium_bg.png",
                    start=data['start'],
                    goal=data['goal'],
                    plan_paths=data['plan_paths'][:frame_idx//replanning_interval+1],
                    ant_path=data['ant_path'][:frame_idx],
                    output_folder=None,
                    index=None,
                    log_to_wandb=False,
                    save_data=False,
                    pos_mean=data['mean'],
                    pos_std=data['std'],
                    last_plan_only=True
                    # orientation_path=data['orientation_path']
                )
                plot_frame = cv2.cvtColor(np.array(plot_frame), cv2.COLOR_BGR2RGB)
            demo_frame = stack_images(render_frame, np.array(plot_frame)[50:-50, 100:-100])
            # cv2.imwrite(os.path.join(video_folder,'frame.png'), demo_frame)
            if video is None:
                dim = tuple(demo_frame.shape[:2][::-1])
                video = cv2.VideoWriter(video_output_path, fourcc, 30, dim)
            video.write(demo_frame)

        if video is not None:
            video.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create demo videos for visualisation')
    parser.add_argument('directory', type=str, help='Directory containing viz outputs')
    args = parser.parse_args()
    generate_demo_video(args.directory)