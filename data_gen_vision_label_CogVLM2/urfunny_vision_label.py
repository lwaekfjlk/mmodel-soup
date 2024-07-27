import os
import time
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
device = 'cuda:0'

image_folder = "../urfunny_data/data_raw/image_data"
data_folder = "../urfunny_data/data_raw"
output_file = "../urfunny_data/data_gen_output/urfunny_image_only_pred_cogvlm2.jsonl"

batch_size = 2
query = 'Describe this image in detail, and the description should be between 15 to 80 words.'

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


ground_truth_labels = {}
for file_name in ["train_data.json", "val_data.json", "test_data.json"]:
    with open(os.path.join(data_folder, file_name), "r") as f:
        data = json.load(f)
        for key, value in data.items():
            ground_truth_labels[key] = value["label"]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

yes_token = tokenizer.tokenize("Yes")
yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
no_token = tokenizer.tokenize("No")
no_token_id = tokenizer.convert_tokens_to_ids(no_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
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
        outputs = model(**input_batch)
        logits = outputs.logits.to("cpu").float()
        for i in range(len(logits)):
            
            last_logits = logits[i, -1, :]
            yes_logits = last_logits[yes_token_id]
            no_logits = last_logits[no_token_id]
            response = {"Yes": yes_logits[0].item(), "No": no_logits[0].item()}

            with open(output_file, "a") as f:
                result = {"image_id": image_id_list[i], 
                          "logits": response, 
                          "gth": ground_truth_labels[image_id_list[i]], 
                          "pred": 1 if yes_logits[0].item() > no_logits[0].item() else 0}
                f.write(json.dumps(result) + "\n")