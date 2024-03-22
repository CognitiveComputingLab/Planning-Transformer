import cv2
import os
import argparse
import re
from tqdm import tqdm

def resize_and_overlay(attention_path, render_path, output_path, height_ratio):
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
    new_height = int(render_height * height_ratio)
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

def main(directory, height_ratio):
    pattern = r'attention_(t=\d+-ep=\d+)\.mp4'
    attention_files = {re.search(pattern, f).groups()[0]:f for f in os.listdir(directory) if re.match(pattern, f)}
    pattern = r'render_(t=\d+-ep=\d+)\.mp4'
    render_files = {re.search(pattern, f).groups()[0]: f for f in os.listdir(directory) if re.match(pattern, f)}

    for suffix,attention_file in attention_files.items():
        if suffix in render_files:
            resize_and_overlay(os.path.join(directory, attention_file),
                               os.path.join(directory, render_files[suffix]),
                               os.path.join(directory, f'combined_{suffix}.mp4'), height_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine and resize videos.')
    parser.add_argument('directory', type=str, help='Directory containing videos')
    parser.add_argument('--height_ratio', type=float, help='Ratio of attention video to render video height')
    args = parser.parse_args()
    main(args.directory, args.height_ratio)