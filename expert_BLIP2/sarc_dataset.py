import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset


class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, image_processor, max_length=128):
        self.dataset = load_dataset("csv", data_files=dataset_path)['train']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        image_path = f'/root/bak/CogVLM/sarc_data/image_data/{item[" image"].strip()}'
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        label = torch.tensor(item[' label'], dtype=torch.long)

        full_prompt = f"Question: The tweet for this image is: {text}. Is the tweet sarcastic (yes or no)? Answer:"
        text_encoding = self.tokenizer(full_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
            "tweet": text,
            "prompt": full_prompt
        }

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