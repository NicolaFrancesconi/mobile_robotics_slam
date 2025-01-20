import os
import imageio

import numpy as np

from PIL import Image
import imageio
import os


path = __file__
file_location_subfolders = 3  # Number of folders to go up to reach the root of the package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)

def _generate_gif():
    # Create GIF from saved frames
    frames = []
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])

    # Ensure all frames have the same size
    target_size = None  # Will store the size of the first image (width, height)
    for frame_file in frame_files:
        img = Image.open(frame_file)
        if target_size is None:
            target_size = img.size  # Set target size to the size of the first image
        else:
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # Resize to match target size
        frames.append(np.array(img))  # Convert to NumPy array for `imageio`

    if frames:
        imageio.mimsave(gif_path, frames, duration=duration)
        print(f"GIF saved to {gif_path}")
    else:
        print("No frames to create a GIF.")

frames_dir = os.path.join(path, "frames")
gif_path = os.path.join(path, "gif","MapRealDingoAuditoriumprova.gif")  # Path for the final GIF
duration = 0.1  # Duration of each frame in seconds

# Create frames directory if it doesn't exist
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Create GIF from saved frames

_generate_gif()


