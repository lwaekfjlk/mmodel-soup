from openai import OpenAI
import base64
import os
import json
import time
from tqdm import tqdm

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_ids_and_image_paths(image_folder_path):
    image_paths = []
    ids = []
    for file in os.listdir(image_folder_path):
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(image_folder_path, file))
            ids.append(file.split(".")[0])
    return ids, image_paths

client = OpenAI()
image_folder_path = "../mustard_data/data_raw/images"
ids, image_paths = get_ids_and_image_paths(image_folder_path)

image_description = {}

for image_path, id in tqdm(zip(image_paths, ids), total=len(ids)):
    base64_image = encode_image(image_path)
    attempts = 0
    success = False
    while attempts < 5 and not success:
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the body language, figurative language, face emotion together with their scenario for characters in the TV show screenshot briefly."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            description = response.choices[0].message.content
            image_description[id] = description
            print(description)
            print('='*50)
            success = True
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts < 5:
                time.sleep(5)
            else:
                print("Failed after 5 attempts.")

        if success:
            break

    with open("../mustard_data/data_gen_output/mustard_image_description.json", "w") as f:
        json.dump(image_description, f)

