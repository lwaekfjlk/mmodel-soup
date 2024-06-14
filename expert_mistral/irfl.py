import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class IRFLDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, max_length=128):
        self.dataset = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return [
            {     "id": id,
                "image_id": id,
                "text": data["text"],
                "label": 1 if 'Figurative' in data['category'] else 0
            }
            for id, data in raw_dataset.items()
        ]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        id = item['id']
        label = torch.tensor(item['label'], dtype=torch.long)
        with open('irfl_captions.json', 'r') as json_file:
            parsed_results = json.load(json_file)
        img_key = item["image_id"] + ".jpeg"
        caption = parsed_results[img_key]
        full_prompt = (
            f"Text:{text}. Image Caption:{caption}"
            f"Do the text and image represent figurative (yes or no)? Answer:"
        )
        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "label": label,
            "id": id
        }

    def tokenize_and_left_pad(self, full_prompt, max_length):
        text_encoding = self.tokenizer(full_prompt, truncation=True, max_length=max_length, return_tensors="pt")
        seq_len = text_encoding['input_ids'].size(1)
        padding_length = max_length - seq_len
        
        if padding_length > 0:
            pad_ids = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((1, padding_length), dtype=torch.long)
            text_encoding['input_ids'] = torch.cat([pad_ids, text_encoding['input_ids']], dim=1)
            text_encoding['attention_mask'] = torch.cat([pad_mask, text_encoding['attention_mask']], dim=1)
        else:
            text_encoding['input_ids'] = text_encoding['input_ids'][:, :max_length]
            text_encoding['attention_mask'] = text_encoding['attention_mask'][:, :max_length]
        return text_encoding


def irfl_collate(batch):
    """
    A custom collate function to pad the batches dynamically.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    ids = [item["id"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "label": labels,
        "id": ids,
    }


def get_irfl_dataloader(args, tokenizer, split):
    if split == "train":
        dataset = IRFLDataset(args.train_path, tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=irfl_collate)
    elif split == "val":
        dataset = IRFLDataset(args.val_path, tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=irfl_collate)
    elif split == "test":
        dataset = IRFLDataset(args.test_path, tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=irfl_collate)
