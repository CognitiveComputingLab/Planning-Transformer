import torch
import torchvision.utils as vutils
import wandb
from PIL import Image
import matplotlib
import numpy as np
import io
import cv2
import pickle
import os
from shapely.geometry import LineString
import math

matplotlib.use('Agg')
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
    np_tensor = tensor.detach().cpu().numpy()

    # Normalize the tensor for better color mapping
    norm_tensor = (np_tensor - np.min(np_tensor)) / (np.max(np_tensor) - np.min(np_tensor))

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.bar(range(len(norm_tensor)), np.ones_like(norm_tensor), color=plt.cm.viridis(norm_tensor))

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
    video_size = tuple(int(max(grid.shape[i] for grid in grids) * scale_factor) for i in [1, 0])

    # Initialize VideoWriter with dynamic video size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, video_size)

    for grid in grids:
        # if we log arrays with shape 1 or 2 in the last dimension we need to add more channnels to bring it to 3
        grid = np.concatenate((grid, np.zeros(grid.shape[:2] + (3 - grid.shape[2],),dtype=np.uint8)), axis=2)
        frame = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)  # Convert color space
        if scale_factor!=1: frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_NEAREST)

        # Calculate padding to center the frame in the video
        pad_width = (video_size[0] - frame.shape[1]) // 2
        pad_height = (video_size[1] - frame.shape[0]) // 2

        # Apply padding
        if pad_width>0 or pad_height>0:
            frame = cv2.copyMakeBorder(frame, pad_height, pad_height, pad_width, pad_width,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

        out.write(frame)  # Write padded frame to the video

    out.release()  # Release the VideoWriter


def normalise_maze_coords(X,mean,std):
    return (np.array(X)-mean)/std

def plot_and_log_paths(image_path, start, goal, plan_paths, ant_path, output_folder, index, pos_mean, pos_std,
                       log_to_wandb=True, save_data=True):
    # if the output folder doesn't exist make it
    os.makedirs(output_folder, exist_ok=True)
    # Load the background image
    bg_image = plt.imread(image_path)

    # Create figure and axes
    fig, ax = plt.subplots()
    tl, br = normalise_maze_coords([[-6,-6], [26,26]],pos_mean, pos_std)
    ax.imshow(bg_image,extent=(tl[0],br[0],tl[1], br[1]))

    goal= np.array(goal)
    # print(goal/8)

    # Plot plan path with blue line and dots
    for plan_path in plan_paths:
        if plan_path.shape[0]:
            # print(plan_path.shape, plan_path)
            ax.plot(plan_path[:, 0], plan_path[:, 1], 'bo-', linewidth=2, markersize=5)

    if ant_path is not None:
        # Plot ant path with rainbow line
        ax.scatter(ant_path[:, 0], ant_path[:, 1], c=np.linspace(0, 1, len(ant_path)), cmap='rainbow', s=2)
        # To create a gradient line, plot each segment in a loop with colors from the 'rainbow' colormap
        norm = plt.Normalize(0, 1)
        cmap = plt.get_cmap('rainbow')
        for i in range(len(ant_path) - 1):
            plt.plot(ant_path[i:i + 2, 0], ant_path[i:i + 2, 1], color=cmap(norm(i / (len(ant_path) - 2))),
                     linewidth=2)

    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=10)  # Start in green
    ax.scatter(goal[0], goal[1], s=100, c='silver', marker='*', zorder=5)  # Goal in silver

    # Remove axes for better visualization
    ax.axis('off')

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
                 'std': pos_std}, f)

def adjust_path_points(simplified_points, target_points):
    while len(simplified_points) < target_points:
        # Find indices of the furthest consecutive points
        furthest_idx = max(range(len(simplified_points) - 1),
                           key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Insert a midpoint
        midpoint = tuple(np.mean([simplified_points[furthest_idx], simplified_points[furthest_idx + 1]], axis=0))
        simplified_points.insert(furthest_idx + 1, midpoint)

    while len(simplified_points) > target_points:
        # Find indices of the closest consecutive points
        closest_idx = min(range(len(simplified_points) - 1),
                          key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Remove one of the closest points
        del simplified_points[closest_idx]

def simplify_path_to_target_points(path, target_points, tolerance=0.1, tolerance_increment=0.05):
    if len(path) < 2 or target_points >= len(path):
        return path

    line = LineString(path)
    simplified_line = line.simplify(tolerance)
    simplified_points = list(simplified_line.coords)

    while len(simplified_points) != target_points:
        if len(simplified_points) > target_points:
            tolerance += tolerance_increment
        else:
            tolerance -= tolerance_increment
            tolerance_increment /= 2

        simplified_line = line.simplify(tolerance)
        simplified_points = list(simplified_line.coords)

        if tolerance <= 0 or tolerance_increment < 1e-5:
            break

    # Directly adjust the number of points to match the target
    adjust_path_points(simplified_points, target_points)

    return simplified_points
