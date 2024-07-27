ort os
import shutil
from tqdm import tqdm

target_dir = "./images/"
for subdir in tqdm(os.listdir("./key_frames/")):
    for file in os.listdir("key_frames/" + subdir):
        # import pdb; pdb.set_trace()
        if file.endswith(('.png')):
            shutil.copy("key_frames/" + subdir + "/" + file, os.path.join(target_dir, f"{subdir}.png"))