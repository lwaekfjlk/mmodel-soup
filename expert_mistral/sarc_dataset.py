import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import json
import ipdb

class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, max_length=128):
        self.dataset = load_dataset("csv", data_files=dataset_path)['train']
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        input_json_file = "sarc_image_descriptions.json"
        # Read the dictionary from the JSON file
        with open(input_json_file, "r") as json_file:
            image_description_dict = json.load(json_file)
        ipdb.set_trace()
        key = item[" image"]
        caption = image_description_dict[key]
        label = torch.tensor(item[' label'], dtype=torch.long)

        full_prompt = f"Question: The description for a tweeted image is {caption}. The tweet text is {text}. Is the tweet sarcastic (yes or no)? Answer:"
        # right padding
        #text_encoding = self.tokenize(full_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        # left padding
        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "label": label,
            "tweet": text,
            "prompt": full_prompt
        }

    def tokenize_and_left_pad(self, full_prompt, max_length):
        # Tokenize without padding
        text_encoding = self.tokenizer(full_prompt, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Calculate necessary padding
        seq_len = text_encoding['input_ids'].size(1)
        padding_length = max_length - seq_len
        
        if padding_length > 0:
            # Create padding tensors
            pad_ids = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((1, padding_length), dtype=torch.long)
            
            # Apply left padding
            text_encoding['input_ids'] = torch.cat([pad_ids, text_encoding['input_ids']], dim=1)
            text_encoding['attention_mask'] = torch.cat([pad_mask, text_encoding['attention_mask']], dim=1)
        else:
            # If no padding is necessary, ensure the sequence is truncated to max_length
            text_encoding['input_ids'] = text_encoding['input_ids'][:, :max_length]
            text_encoding['attention_mask'] = text_encoding['attention_mask'][:, :max_length]

        return text_encoding

def custom_collate(batch):
    """
    A custom collate function to pad the batches dynamically.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "image": images,
        "label": labels
    }