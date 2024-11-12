import os
import shutil
from math import ceil

from tqdm import tqdm

# Define the path to your folder containing images
folder_path = "../sarc_data/data_raw/images"

# Get a list of all files in the folder
files = [
    f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
]

# Calculate the number of files per subfolder
num_files = len(files)
files_per_subfolder = ceil(num_files / 4)

# Create subfolders
for i in range(4):
    subfolder_path = os.path.join(folder_path, f"subfolder_{i+1}")
    os.makedirs(subfolder_path, exist_ok=True)

# copy files to subfolders
for i, file in enumerate(tqdm(files)):
    subfolder_index = i // files_per_subfolder
    subfolder_path = os.path.join(folder_path, f"subfolder_{subfolder_index+1}")
    shutil.copy(os.path.join(folder_path, file), os.path.join(subfolder_path, file))

print("Files have been split into 4 subfolders.")
