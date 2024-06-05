import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class NYCartoonDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, image_data_path, tokenizer, image_processor, max_length=128):
        self.dataset = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self.image_data_path = image_data_path

    def __len__(self):
        return len(self.dataset)
    
    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return [
            {
                "id": id,
                "image_id": id.split('_')[0],
                "caption": data["caption"],
                "label": data['label'],
                "question": data["questions"],
            }
            for id, data in raw_dataset.items()
        ]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        caption = item['caption']
        question = item['question']
        id = item['id']
        image_path = f'{self.image_data_path}/{item["image_id"]}.jpg'
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)

        full_prompt = (
            f"Do the caption of {caption} and image represent humor or match with each other (yes or no)? The question is {question} Answer:",
        )

        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
            "id": id,
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


def nycartoon_collate(batch):
    """
    A custom collate function to pad the batches dynamically.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    ids = [item["id"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "image": images,
        "label": labels,
        "id": ids,
    }


def get_nycartoon_dataloader(args, tokenizer, image_processor, split):
    if split == "train":
        dataset = NYCartoonDataset(args.train_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=nycartoon_collate)
    elif split == "val":
        dataset = NYCartoonDataset(args.val_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=nycartoon_collate)
    elif split == "test":
        dataset = NYCartoonDataset(args.test_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=nycartoon_collate)
