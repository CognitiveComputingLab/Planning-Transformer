import torch
import torchvision.utils as vutils
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io

def log_attention_maps(attention_maps, log_key="attention_map_grid"):
    num_heads = attention_maps[0].size(1)
    num_layers = len(attention_maps)
    batch_size = attention_maps[0].size(0)

    # Limit the batch size to 10 if larger
    batch_limit = min(batch_size, 10)

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

        # Log the grid image to wandb for this head
        wandb.log({f"{log_key}_head_{head}": wandb.Image(grid_image)})

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