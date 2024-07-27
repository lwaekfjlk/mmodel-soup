import os
import time
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
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


def generate(image_folder, output_file, batch_size, query, model_path, torch_type, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_type,
        trust_remote_code=True,
        device_map=device,
    ).eval()

    data = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                data.append({"image": os.path.join(root, file)})

    length = len(data)

    for idx in tqdm(range(0, length, batch_size)):
        i_list = []
        for i in range(batch_size):
            if idx + i < length:
                i_list.append(data[idx + i])
            else:
                break
        
        image_id_list = []
        input_sample_list = []
        start = time.time()
        for i in i_list:
            image = Image.open(i["image"]).convert('RGB')
            input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
            input_sample_list.append(input_sample)
            image_id_list.append(i["image"].split("/")[-1].split(".")[0])
        print(f"Prepare input time: {time.time() - start}")

        start = time.time()
        input_batch = collate_fn(input_sample_list, tokenizer)
        input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
        input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
        print(f"Prepare batch time: {time.time() - start}")

        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(outputs)


        outlist = [output.split("<|end_of_text|>")[0].strip() for output in outputs]
        print(f"Generate time: {time.time() - start}")
        
        with open(output_file, "a") as f:
            for i, out in enumerate(outlist):
                result = {"image": image_id_list[i], "caption": out}
                f.write(json.dumps(result) + "\n")