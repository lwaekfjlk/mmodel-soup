import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MustardDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, image_data_path, tokenizer, image_processor, max_length=512):
        dataset_dict = {
            "R": "../mustard_data/data_split_output/mustard_R_dataset_train.json",
            "U": "../mustard_data/data_split_output/mustard_U_dataset_train.json",
            "S": "../mustard_data/data_split_output/mustard_AS_dataset_train.json",
        }
        self.dataset = self.load_dataset(dataset_dict)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_data_path = image_data_path
        self.max_length = max_length

    def load_dataset(self, dataset_dict):
        label_map = {"R": 0, "U": 1, "S": 2}
        overall_dataset = []

        for type, file_path in dataset_dict.items():
            with open(file_path) as f:
                data = json.load(f)
            for id, content in data.items():
                overall_dataset.append({
                    "id": id,
                    "image_id": id,
                    "show": content["show"],
                    "context": content["context"],
                    "speaker": content["speaker"],
                    "utterance": content["utterance"],
                    "label": label_map[type]
                })
        return overall_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = f'{self.image_data_path}/{item["image_id"]}.jpg'
        image = self.image_processor(Image.open(image_path), return_tensors="pt").pixel_values.squeeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        full_prompt = (
            f"Question: You are watching an episode of {item['show']}. "
            f"Following up to this image input, the dialogue has been {item['context']}. "
            f"Given the current speaker is {item['speaker']} who says {item['utterance']} - "
            f"are they being sarcastic? Answer:"
        )

        text_encoding = self.tokenizer(full_prompt, truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt")
        
        return {
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
            "id": item["id"],
        }

def mustard_collate(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "id": [item["id"] for item in batch],
    }

def get_mustard_dataloader(args, tokenizer, image_processor, split):
    dataset = MustardDataset(args.image_data_path, tokenizer, image_processor, args.max_length)
    batch_size = args.batch_size if split == "train" else args.val_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=mustard_collate)