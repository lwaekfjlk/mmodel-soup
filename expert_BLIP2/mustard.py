import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MustardDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, image_data_path, tokenizer, image_processor, max_length=512):
        self.dataset = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor
        self.image_data_path = image_data_path

    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return [
            {
                "id": id,
                "image_id": id,
                "show": data["show"],
                "context": data["context"],
                "speaker": data["speaker"],
                "utterance": data["utterance"],
                "label": data["sarcasm"]
            } 
            for id, data in raw_dataset.items()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        id = item['id']
        image_path = f'{self.image_data_path}/{item["image_id"]}.jpg'
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        label = torch.tensor(item['label'], dtype=torch.long)
        full_prompt = (
            f"Question: You are watching an episode of {item['show']}. "
            f"Following up to this image input, the dialogue has been {item['context']}. "
            f"Given the current speaker is {item['speaker']} who says {item['utterance']} - "
            f"are they being sarcastic? Answer:"
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


def mustard_collate(batch):
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


def get_mustard_dataloader(args, tokenizer, image_processor, split):
    if split == "train":
        dataset = MustardDataset(args.train_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mustard_collate)
    elif split == "val":
        dataset = MustardDataset(args.val_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=mustard_collate)
    elif split == "test":
        dataset = MustardDataset(args.test_path, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=mustard_collate)
