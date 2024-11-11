import os
import time
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
from utils import recur_move_to, collate_fn, generate

image_folder = "../mustard_data/data_raw/images"
output_file = "../mustard_data/data_raw/image_captions_cogvlm2.jsonl"

batch_size = 12
query = (
    "Describe the image in detail."
    "If there are people, focus on their emotions, postures, facial expressions, body language, and interactions. Based on these information, infer what is the event going on."
    "If there are no people, analyze the event or scene, considering background elements and overall context to infer what is the event going on."
    "Provide evidence to predict if the situation is sarcastic."
    "Ensure the description is between 15 to 100 words."
)

model_path = "./cogvlm2-llama3-chat-19B"
torch_type = torch.bfloat16
device = 'cuda:0'


if __name__ == '__main__':
    generate(image_folder, output_file, batch_size, query, model_path, torch_type, device)
