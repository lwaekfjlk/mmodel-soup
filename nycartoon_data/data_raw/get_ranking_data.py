from datasets import load_dataset
from tqdm import tqdm
import json
import ipdb

def convert_string(s):
    return int(ord(s) - ord('A'))

dataset_name = "jmhessel/newyorker_caption_contest"
dataset = load_dataset(dataset_name, 'ranking')
splits = ['train', 'validation', 'test']


description_dataset = {}
for split in splits:
    formatted_dataset = {}
    dataset_split = dataset[split]
    for data in tqdm(dataset_split):
        image = data['image']
        instance_id = data['instance_id']
        caption_choices = data['caption_choices']
        description = data['image_description']
        # save images
        image = data['image']
        image.save(f"images/{instance_id}.jpg")
        del data['image']
        # ABCDE index
        label = data['label']
        for idx, caption in enumerate(caption_choices):
            correct_id = convert_string(label)
            formatted_dataset[instance_id + '_' + caption] = {
                'caption': caption,
                'label': 1 if idx == correct_id else 0,
                'questions': data['questions'],
                'description': description,
                'mode': "single"
            }
        
        image_description = data['image_description']
        image_uncanny_description = data['image_uncanny_description']
        image_location = data['image_location']
        description_dataset[instance_id] = {
            'image_description': image_description,
            'image_uncanny_description': image_uncanny_description,
            'image_location': image_location,
        }

    with open(f"nycartoon_dataset_ranking_{split}.json", 'w') as f:
        json.dump(formatted_dataset, f, indent=4)

with open(f"nycartoon_image_description_ranking.json", 'w') as f:
    json.dump(description_dataset, f, indent=4)
        