import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset
import json


class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, split, max_length=512):
        self.dataset = load_dataset(dataset_path, "matching")[split]
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        choices = item['caption_choices']
        caption_a, caption_b, caption_c, caption_d, caption_e  = choices[0], choices[1],choices[2],choices[3], choices[4]
        labelTranslate = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        label = labelTranslate[item['label']]
        description = item['image_description']
        label = torch.tensor(label, dtype=torch.long)
        full_prompt = f"Question: You are given an image description and 5 caption choices. Description: {description}. 1: {caption_a}. 2: {caption_b}. 3: {caption_c}. 4: {caption_d}. 5: {caption_e}. Can you return which choice suits the image best? Answer:"
        # right padding
        #ipdb.set_trace()s
        #text_encoding = self.tokenize(full_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        # left padding
        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "label": label,
            "prompt": full_prompt}

    def tokenize_and_left_pad(self, full_prompt, max_length):
        text_encoding = self.tokenizer(full_prompt, truncation=True, max_length=max_length, return_tensors="pt")
        
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
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "label": labels
    }