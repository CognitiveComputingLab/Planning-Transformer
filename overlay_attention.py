import cv2
import os
import argparse
import re
from tqdm import tqdm

def resize_and_overlay(attention_path, render_path, output_path):
    # Load videos
    attention_cap = cv2.VideoCapture(attention_path)
    render_cap = cv2.VideoCapture(render_path)

    # Get properties of the render video
    render_width = int(render_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    render_height = int(render_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = render_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(render_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new dimensions for attention video
    attention_height = int(attention_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    attention_width = int(attention_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(render_height / 3)
    new_width = int(attention_width * new_height / attention_height)

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (render_width, render_height))

    # Padding
    pad_height = int(render_height * 0.02)
    pad_width = pad_height

    with tqdm(total=total_frames, desc=f'Processing {os.path.basename(output_path)}') as pbar:
        while True:
            ret1, attention_frame = attention_cap.read()
            ret2, render_frame = render_cap.read()

            if not ret1 or not ret2:
                break

            # Resize attention video
            attention_frame_resized = cv2.resize(attention_frame, (new_width, new_height),interpolation = cv2.INTER_NEAREST)
            t = render_height-pad_height-new_height
            b = render_height-pad_height
            l = render_width-pad_width-new_width
            r = render_width-pad_width
            # Overlay attention video on render video
            render_frame[t:b,l:r] = attention_frame_resized

            out.write(render_frame)
            pbar.update(1)

    attention_cap.release()
    render_cap.release()
    out.release()

def main(directory):
    attention_files = [f for f in os.listdir(directory) if re.match(r'attention_t=\d+\.mp4', f)]
    render_files = {re.search(r'(\d+)', f).group(0): f for f in os.listdir(directory) if re.match(r'render_t=\d+\.mp4', f)}

    for attention_file in attention_files:
        num = re.search(r'(\d+)', attention_file).group(0)
        if num in render_files:
            resize_and_overlay(os.path.join(directory, attention_file),
                               os.path.join(directory, render_files[num]),
                               os.path.join(directory, f'combined_t={num}.mp4'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine and resize videos.')
    parser.add_argument('directory', type=str, help='Directory containing videos')
    args = parser.parse_args()
    main(args.directory)