from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import os

# Extract the ZIP file
zip_folder = './visualisations/PDTv2-oracle-plan/antmaze-medium-diverse-v2/run_36'
zip_path = os.path.join(zip_folder,'path_t=135000.zip')
extract_folder = zip_path.replace('.zip','')

with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Get the first 10 images from the newly extracted folder
extracted_files_new = os.listdir(extract_folder)
images_paths_new = [os.path.join(extract_folder, file) for file in extracted_files_new if file.lower().endswith(('.png', '.jpg', '.jpeg'))][:9]

# Function to trim borders of an image
def trim_borders_new(image_path, border_size):
    with Image.open(image_path) as img:
        # Crop the image
        cropped_img = img.crop((border_size, border_size, img.width - border_size, img.height - border_size))
        return cropped_img

# Trim borders of each new image
trimmed_images_new = [trim_borders_new(img_path, 20) for img_path in images_paths_new]

# Display the trimmed images in a 3x3 grid
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for ax, img in zip(axs.ravel(), trimmed_images_new):
    img = img.convert("RGBA")  # Convert images to a consistent format for matplotlib
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()