import os
import time
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        return item.to(tgt)
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
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

def initialize_model_and_tokenizer(model_path, torch_type, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_type,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    return model, tokenizer

def prepare_data(image_folder):
    data = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                data.append({"image": os.path.join(root, file)})
    return data

def process_batch(data_batch, model, tokenizer, device, query):
    image_id_list = []
    input_sample_list = []
    for item in data_batch:
        image = Image.open(item["image"]).convert('RGB')
        input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
        input_sample_list.append(input_sample)
        image_id_list.append(item["image"].split("/")[-1].split(".")[0])

    input_batch = collate_fn(input_sample_list, tokenizer)
    input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
    input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
    return input_batch, image_id_list

def save_results(output_file, image_id_list, captions):
    with open(output_file, "a") as f:
        for image_id, caption in zip(image_id_list, captions):
            result = {"image": image_id, "caption": caption}
            f.write(json.dumps(result) + "\n")