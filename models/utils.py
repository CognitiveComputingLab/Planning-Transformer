import torch
import torchvision.utils as vutils
import wandb
from PIL import Image
import matplotlib
import numpy as np
import io
import cv2

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

def log_tensor_as_image(tensor, log_key="tensor_visualization"):
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

    # Log the image to wandb
    wandb.log({log_key: wandb.Image(img)})

def arrays_to_video(arrays, output_file, scale_factor=1.0, fps=30.0):
    """
    Converts a list of NumPy arrays into a video, with an option to scale the frames.
    Adjusts frames of varying sizes to the largest frame size in the list.

    Parameters:
    arrays (list of np.ndarray): List of NumPy arrays representing the frames.
    output_file (str): Path to the output video file.
    scale_factor (float): Factor to scale the frames. Default is 1.0 (no scaling).
    fps (float): Frames per second of the output video. Default is 30.0.
    """
    # Find the maximum frame size
    max_width = max(arr.shape[1] for arr in arrays)
    max_height = max(arr.shape[0] for arr in arrays)
    video_size = (int(max_width * scale_factor), int(max_height * scale_factor))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_file, fourcc, fps, video_size)

    for arr in arrays:
        #fix colour
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Scale the frame
        scaled_frame = cv2.resize(arr, (int(arr.shape[1] * scale_factor), int(arr.shape[0] * scale_factor)),
                                  interpolation = cv2.INTER_NEAREST)

        # Pad the frame with black pixels to match the video size
        pad_width = (video_size[0] - scaled_frame.shape[1]) // 2
        pad_height = (video_size[1] - scaled_frame.shape[0]) // 2

        padded_frame = cv2.copyMakeBorder(scaled_frame, pad_height, pad_height, pad_width, pad_width,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Write the frame
        out.write(padded_frame)

    # Release everything when job is finished
    out.release()