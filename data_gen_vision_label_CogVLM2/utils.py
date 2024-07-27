import os
import torch
import json
from PIL import Image
from transformers import AutoTokenizer

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        return item.to(tgt)
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple(recur_move_to(v, tgt, criterion_func) for v in item)
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)

    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature

    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }

    if images:
        batch['images'] = images

    return batch

def load_ground_truth_labels(data_folder, file_names):
    ground_truth_labels = {}
    for file_name in file_names:
        with open(os.path.join(data_folder, file_name), "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if 'sarcasm' in value:
                    ground_truth_labels[key] = 1 if value["sarcasm"] else 0
                else:
                    ground_truth_labels[key] = value["label"]
    return ground_truth_labels

def load_images(image_folder, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    data = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(extensions):
                data.append({"image": os.path.join(root, file)})
    return data

def prepare_input_samples(model, tokenizer, query, image_list):
    input_sample_list = []
    image_id_list = []
    for image_path in image_list:
        image = Image.open(image_path).convert('RGB')
        input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
        input_sample_list.append(input_sample)
        image_id_list.append(image_path.split("/")[-1].split(".")[0])
    return input_sample_list, image_id_list

def save_results(output_file, image_id, response, ground_truth_label, prediction):
    with open(output_file, "a") as f:
        result = {"image_id": image_id, "logits": response, "gth": ground_truth_label, "pred": prediction}
        f.write(json.dumps(result) + "\n")
