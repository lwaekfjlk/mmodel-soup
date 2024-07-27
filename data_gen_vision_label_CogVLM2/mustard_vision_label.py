import os
import time
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import recur_move_to, collate_fn, initialize_model_and_tokenizer, prepare_data, save_results

MODEL_PATH = "/dataset/granite_ckpt/haofeiyu/cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
device = 'cuda:0'

image_folder = "../mustard_data/data_raw/images"
output_file = "../mustard_data/data_gen_output/mustard_vision_only_pred_cogvlm2.json"

batch_size = 3
query = 'Describe this image in detail, and the description should be between 15 to 80 words.'


def predict(model, input_batch, tokenizer):
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }
    with torch.no_grad():
        outputs = model.generate(**input_batch, **gen_kwargs)
        outputs = outputs[:, input_batch['input_ids'].shape[1]:]
        outputs = tokenizer.batch_decode(outputs)
    return [output.split("")[0].strip() for output in outputs]


def main():
    model, tokenizer = initialize_model_and_tokenizer(MODEL_PATH, TORCH_TYPE, device)
    data = prepare_data(image_folder)
    length = len(data)
    
    for idx in tqdm(range(0, length, batch_size)):
        data_batch = data[idx:idx+batch_size]
        input_batch, image_id_list = process_batch(data_batch, model, tokenizer, device, query)
        captions = predict(model, input_batch, tokenizer)
        save_results(output_file, image_id_list, captions)

if __name__ == "__main__":
    main()
