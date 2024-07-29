import io
import math
import os
import pickle

import cv2
import matplotlib
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from scipy.spatial.transform import Rotation as R
import time
import re
import wandb
from tqdm import tqdm
# matplotlib.use('Agg')
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt


def log_attention_maps(attention_maps, log_key="attention_map_grid", log_to_wandb=True):
    num_layers = len(attention_maps)

    # Check if the attention maps have a batch dimension, add one if not
    if attention_maps[0].dim() == 3:  # Assuming shape [num_heads, seq_len, seq_len] if no batch dimension
        attention_maps = [attention_map.unsqueeze(0) for attention_map in attention_maps]  # Add a batch dimension
        batch_size = 1
    else:
        batch_size = attention_maps[0].size(0)

    num_heads = attention_maps[0].size(1)

    # Limit the batch size to 10 if larger
    batch_limit = min(batch_size, 10)
    maps = []
    for head in range(num_heads):
        # Collect the attention maps for this head across all layers and limited batch
        head_maps = torch.cat(
            [attention_maps[layer][:batch_limit, head, :, :].unsqueeze(0) for layer in range(num_layers)], 0)

        # Transpose the tensor to have batches as columns and layers as rows
        head_maps = head_maps.permute(1, 0, 2, 3)

        # Create a grid for this head
        grid = vutils.make_grid(head_maps, nrow=num_layers, normalize=True, scale_each=True)

        # Convert the grid to a PIL Image
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_image = Image.fromarray((grid_np * 255).astype('uint8'))
        maps.append(np.array(grid_image))

        if log_to_wandb:
            # Log the grid image to wandb for this head
            wandb.log({f"{log_key}_head_{head}": wandb.Image(grid_image)})

    return maps


def log_tensor_as_image(tensor, log_key="tensor_visualization", log_to_wandb=True):
    # Ensure the tensor is 1-dimensional
    if tensor.ndim != 1:
        raise ValueError("Input tensor must be 1-dimensional")

    # Convert the tensor to a numpy array
    np_tensor = tensor.numpy()

    # Normalize the tensor for better color mapping
    # np_tensor = (np_tensor - np.min(np_tensor)) / (np.max(np_tensor) - np.min(np_tensor))

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.bar(range(len(np_tensor)), np.ones_like(np_tensor), color=plt.cm.viridis(np_tensor))

    # Remove axes for cleaner visualization
    ax.axis('off')

    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)

    # Close the plt figure to free memory
    plt.close(fig)

    if log_to_wandb:
        # Log the image to wandb
        wandb.log({log_key: wandb.Image(img)})
    return np.array(img)


def create_grid(timesteps, dynamic_padding=True):
    num_images = len(timesteps)
    grid_cols = int(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    img_height, img_width = timesteps[0].shape[:2]
    padding = img_width // 10 if dynamic_padding else 10

    grid_height = img_height * grid_rows + padding * (grid_rows - 1)
    grid_width = img_width * grid_cols + padding * (grid_cols - 1)
    grid = np.zeros((grid_height, grid_width, timesteps[0].shape[-1]), dtype=np.uint8)

    for idx, img in enumerate(timesteps):
        row, col = divmod(idx, grid_cols)
        start_y = row * (img_height + padding)
        start_x = col * (img_width + padding)
        grid[start_y:start_y + img_height, start_x:start_x + img_width] = img

    return grid


def arrays_to_video(timesteps_arrays, output_file, scale_factor=1.0, fps=30.0, use_grid=True):
    # Pre-compute video size based on the largest grid dimensions after scaling
    grids = [create_grid(timestep) for timestep in timesteps_arrays] if use_grid else timesteps_arrays
    grids = [np.array(grid) for grid in grids if len(grid) > 0]
    if len(grids) == 0: return True
    max_dims = [max(grid.shape[i] for grid in grids) for i in [1, 0]]
    video_size = tuple(int(dim * scale_factor) for dim in max_dims)

    # Initialize VideoWriter with dynamic video size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, video_size)

    for i, grid in enumerate(grids):
        # if the image has an alpha channel we'll remove it
        grid = grid[:, :, :3]
        # if we log arrays with shape 1 or 2 in the last dimension we need to add more channnels to bring it to 3
        grid = np.concatenate((grid, np.zeros(grid.shape[:2] + (3 - grid.shape[2],), dtype=np.uint8)), axis=2)
        frame = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)  # Convert color space

        if scale_factor != 1: frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_NEAREST)

        # Calculate padding to center the frame in the video
        pad_width = (video_size[0] - frame.shape[1]) // 2
        pad_height = (video_size[1] - frame.shape[0]) // 2

        # Apply padding
        if pad_width > 0 or pad_height > 0:
            frame = cv2.copyMakeBorder(frame, pad_height, pad_height, pad_width, pad_width,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])

        out.write(frame)  # Write padded frame to the video

    out.release()  # Release the VideoWriter


def normalise_coords(X, mean, std):
    return (np.array(X) - mean) / std


def prepare_plot(image_path, pos_mean, pos_std):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Determine the extent of the image based on its path
    if 'antmaze' in image_path:
        if 'large' in image_path:
            tl, br = normalise_coords([[-6, -6], [42, 30]], pos_mean, pos_std)
        elif 'medium' in image_path:
            tl, br = normalise_coords([[-6, -6], [26, 26]], pos_mean, pos_std)
        else:
            tl, br = normalise_coords([[-6, -6], [14, 14]], pos_mean, pos_std)
    elif 'kitchen' in image_path:
        tl, br = normalise_coords([[-0.32, -0.85], [-0.22, 1.35]], pos_mean, pos_std)
    else:
        tl, br = normalise_coords([[-0.5, -0.5], [-0.5, 0.5]], pos_mean, pos_std)

    # Load the background image
    if image_path is not None:
        bg_image = plt.imread('./visualisations/bg_images/'+image_path)
        ax.imshow(bg_image, extent=(tl[0], br[0], tl[1], br[1]))

    return fig, ax


def plot_and_log_paths(image_path, start, goal, plan_paths, ant_path, output_folder, index, pos_mean, pos_std,
                       log_to_wandb=True, save_data=True, orientation_path=None, last_plan_only=False):
    if output_folder is not None:
        # if the output folder doesn't exist make it
        os.makedirs(output_folder, exist_ok=True)

    fig, ax = prepare_plot(image_path, pos_mean, pos_std)

    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=10)  # Start in green
    if goal is not None:
        goal = np.array(goal)
        ax.scatter(goal[0], goal[1], s=100, c='silver', marker='*', zorder=5)  # Goal in silver

    # Plot plan paths with different coloured lines and dots
    if last_plan_only:
        plan_paths = plan_paths[-1:]
    for i, plan_path in enumerate(plan_paths):
        if (i % 40 != 0 and i != len(plan_paths) - 1 and i != 0):
            continue
        # if not (i == 0 or i == len(plan_paths)-1):
        #     continue
        if plan_path.shape[0]:
            # print(plan_path.shape, plan_path)
            ax.plot(plan_path[:, 0], plan_path[:, 1], '-o', linewidth=2, markersize=4, alpha=0.5)
            ax.plot(plan_path[0, 0], plan_path[0, 1], 'o', color='red', markersize=4, alpha=1.0)

    if ant_path is not None:
        # Plot ant path with rainbow line
        ax.scatter(ant_path[:, 0], ant_path[:, 1], c=np.linspace(0, 1, len(ant_path)), cmap='rainbow', s=2)

    # print([(i, float(f"{rotate_orientation_vectors(o)[1][2]:.1f}")) for i,o in enumerate(orientation_path)])
    if orientation_path is not None:
        orientation = orientation_path[-1]
        forward_rot, up_rot, right_rot = rotate_orientation_vectors(orientation)
        has_fallen = up_rot[2] < 0.25
        # print(up_rot, has_fallen)
        if has_fallen:
            ax.scatter(ant_path[-1][0], ant_path[-1][1], s=100, c='yellow', marker='x', zorder=5)  # mark that it fell

    # Remove axes for better visualization
    ax.axis('off')

    if output_folder is not None:
        # Save the image
        output_path = os.path.join(output_folder, f"path_{index}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    # Log image to wandb
    if log_to_wandb:
        wandb.log({f"path_{index}": wandb.Image(output_path)})

    if save_data:
        # Pickle the data
        with open(os.path.join(output_folder, f"path_data_{index}.pkl"), 'wb') as f:
            pickle.dump(
                {'start': start, 'goal': goal, 'plan_paths': plan_paths, 'ant_path': ant_path, 'mean': pos_mean,
                 'std': pos_std, 'orientation_path': orientation_path}, f)

    if output_folder is None:
        fig.canvas.draw()
        output = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return output


def rotate_orientation_vectors(orientation):
    rotation = R.from_euler("zyx", orientation[:3], degrees=False)
    right = np.array([1, 0, 0])
    forward = np.array([0, 1, 0])
    up = np.array([0, 0, 1])
    return rotation.apply(forward), rotation.apply(up), rotation.apply(right)


def plot_orientation_vectors(ax, position, forward_vector, up_vector, right_vector, length=0.5):
    x, y, z = position
    ax.quiver(x, y, z, forward_vector[0], forward_vector[1], forward_vector[2], color='red', length=length)
    ax.quiver(x, y, z, up_vector[0], up_vector[1], up_vector[2], color='green', length=length)
    ax.quiver(x, y, z, right_vector[0], right_vector[1], right_vector[2], color='blue', length=length)


def plot_and_log_paths_3d(image_path, start, goal, plan_paths, ant_path, orientation_path, output_folder, index,
                          pos_mean, pos_std, log_to_wandb=True, save_data=True):
    os.makedirs(output_folder, exist_ok=True)
    bg_image = plt.imread('./visualisations/bg_images/'+image_path)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12, 8))
    ax.view_init(elev=60, azim=-45)

    tl, br = normalise_coords([[-6, -6], [26, 26]], pos_mean[:2], pos_std[:2])
    ax.set_xlim(np.array([tl[0], br[0]]) * 0.5)
    ax.set_ylim(np.array([tl[1], br[1]]) * 0.5)
    ax.set_zlim([0, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot

    # X, Y = np.meshgrid(np.linspace(tl[0], br[0], bg_image.shape[1]), np.linspace(br[1], tl[1], bg_image.shape[0]))  # Use the reversed Y array here
    # ax.plot_surface(X, Y, np.zeros(X.shape), rstride=1, cstride=1, facecolors=plt.imshow(bg_image).get_array(),
    #                 shade=False)

    ax.scatter(start[0], start[1], 1, s=100, c='yellow', marker='o')
    ax.scatter(goal[0], goal[1], 1, s=100, c='silver', marker='*')

    for plan_path in plan_paths:
        if plan_path.shape[0]:
            ax.plot(plan_path[:, 0], plan_path[:, 1], zs=1, zdir='z', marker='o', linewidth=2, markersize=4, alpha=0.5)

    if ant_path is not None:
        # Plot ant path in 3D with rainbow gradient
        norm = plt.Normalize(0, 1)
        cmap = plt.get_cmap('rainbow')
        for i in range(len(ant_path)):
            ax.plot(ant_path[i:i + 2, 0], ant_path[i:i + 2, 1], zs=1, zdir='z',
                    color=cmap(norm(i / (len(ant_path) - 1))), linewidth=2)

    if orientation_path is not None:
        for i, orientation in enumerate(orientation_path):
            if (i % 40 == 0 and i < len(orientation_path) // 2) or i == len(orientation_path) - 1:
                forward_rot, up_rot, right_rot = rotate_orientation_vectors(orientation)
                point = np.concatenate((ant_path[i][:2], np.array([1])))
                plot_orientation_vectors(ax, point, forward_rot, up_rot, right_rot, length=0.4)

    # ax.axis('off')
    output_path = os.path.join(output_folder, f"path_{index}_3D.png")
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    if log_to_wandb:
        wandb.log({f"3D_path_{index}": wandb.Image(output_path)})

    if save_data:
        with open(os.path.join(output_folder, f"path_data_{index}.pkl"), 'wb') as f:
            pickle.dump({'start': start, 'goal': goal, 'plan_paths': plan_paths, 'ant_path': ant_path,
                         'orientation_path': orientation_path, 'mean': pos_mean, 'std': pos_std}, f)

def demo_goal_select(image_path, pos_mean, pos_std):
    fig, ax = prepare_plot(image_path, pos_mean, pos_std)
    plt.title('Click on the image to set a goal')
    print("Please click on the image to set a goal.")
    goal = plt.ginput(1)
    print("Goal set at:", goal)
    ax.plot(goal[0][0], goal[0][1], 'r*', markersize=15)
    plt.show(block=False)
    time.sleep(1)
    plt.close()
    return goal[0]

