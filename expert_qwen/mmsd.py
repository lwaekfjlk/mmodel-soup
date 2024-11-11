import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class mmsdDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer,  max_length=512):
        self.dataset = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return [
            {   "id": id,
                "label": data["label"],
                "text": data["text"]
            } 
            for id, data in raw_dataset.items()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        id = item['id']
        text = item['text']
        label = torch.tensor(item['label'], dtype=torch.long)
        input_json_file = ".../mmsd_data/data_gen_output/sarc_image_description.json"
        # Read the dictionary from the JSON file
        with open(input_json_file, "r") as json_file:
            image_description_dict = json.load(json_file)
        caption = image_description_dict[item['id']]
        full_prompt = f"Question: Given the image captioned as {caption} and the text {text} are they being sarcastic? Answer:"
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


def mmsd_collate(batch):
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


def get_mmsd_dataloader(args, tokenizer,  split):
    if split == "train":
        dataset = mmsdDataset(args.train_path, tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mmsd_collate)
    elif split == "val":
        dataset = mmsdDataset(args.val_path, tokenizer,  args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mmsd_collate)
    elif split == "test":
        dataset = mmsdDataset(args.test_path,  tokenizer,args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mmsd_collate)
