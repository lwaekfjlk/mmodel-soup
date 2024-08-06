import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MustardDataset(Dataset):
    def __init__(self, split, image_data_path, tokenizer, image_processor, max_length=512):
        if split == "train":
            dataset_files = {
                "R": ["../mustard_data/data_split_output/mustard_R_dataset_train_cogvlm2_qwen2_for_fuser.json", "../mustard_data/data_split_output/mustard_R_dataset_val_cogvlm2_qwen2_for_fuser.json"],
                "U": ["../mustard_data/data_split_output/mustard_U_dataset_train_cogvlm2_qwen2_for_fuser.json", "../mustard_data/data_split_output/mustard_U_dataset_val_cogvlm2_qwen2_for_fuser.json"],
                "S": ["../mustard_data/data_split_output/mustard_AS_dataset_train_cogvlm2_qwen2_for_fuser.json", "../mustard_data/data_split_output/mustard_AS_dataset_val_cogvlm2_qwen2_for_fuser.json"],
            }
        elif split == "val":
            dataset_files = {
                "R": ["../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2_for_fuser.json"],
                "U": ["../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2_for_fuser.json"],
                "S": ["../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2_for_fuser.json"],
            }
        elif split == "test":
            dataset_files = {
                "R": ["../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2_for_fuser.json"],
                "U": ["../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2_for_fuser.json"],
                "S": ["../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2_for_fuser.json"],
            }

        with open("../mustard_data/data_raw/mustard_dataset_test.json") as f:
            self.gth = {id: 1 if data["sarcasm"] else 0 for id, data in json.load(f).items()}
        
        with open("../mustard_data/data_raw/mustard_dataset_train.json") as f:
            self.gth.update({id: 1 if data["sarcasm"] else 0 for id, data in json.load(f).items()})
        
        with open("../mustard_data/data_raw/mustard_dataset_val.json") as f:
            self.gth.update({id: 1 if data["sarcasm"] else 0 for id, data in json.load(f).items()})

        self.dataset = self.load_dataset(dataset_files)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_data_path = image_data_path
        self.max_length = max_length

    def load_dataset(self, dataset_files):
        overall_dataset = []
        label_map = {"R": 0, "U": 1, "S": 2}
        
        for type, file_paths in dataset_files.items():
            for file_path in file_paths:
                with open(file_path) as f:
                    data = json.load(f)
                for id, content in data.items():
                    overall_dataset.append({
                        "id": id,
                        "image_id": id,
                        "text": content["utterance"],
                        "rus_label": label_map[type],
                        "task_label": self.gth[id],
                    })
        return overall_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = f'{self.image_data_path}/{item["image_id"]}.jpg'
        image = self.image_processor(Image.open(image_path), return_tensors="pt").pixel_values.squeeze(0)
        rus_label = torch.tensor(item['rus_label'], dtype=torch.long)
        task_label = torch.tensor(item['task_label'], dtype=torch.long)
        text_encoding = self.tokenizer(
            item['text'],
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt",
            padding='longest'
        )
        
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "rus_label": rus_label,
            "task_label": task_label,
            "id": item["id"],
        }

def mustard_collate(batch):
    # pad the input_ids and attention_mask to the same length, adding padding to the left side
    max_length = max(len(item["input_ids"]) for item in batch)
    
    for item in batch:
        padding_length = max_length - len(item["input_ids"])
        if padding_length > 0:
            pad_ids = torch.full((padding_length,), 0, dtype=torch.long)
            pad_mask = torch.zeros((padding_length,), dtype=torch.long)
            item["input_ids"] = torch.cat([pad_ids, item["input_ids"]], dim=0)
            item["attention_mask"] = torch.cat([pad_mask, item["attention_mask"]], dim=0)
    
    # Ensure image tensors have the same dimensions
    max_image_shape = tuple(max(s) for s in zip(*[item["image"].shape for item in batch]))
    for item in batch:
        pad_image = torch.zeros(max_image_shape, dtype=item["image"].dtype)
        pad_image[:item["image"].shape[0], :item["image"].shape[1], :item["image"].shape[2]] = item["image"]
        item["image"] = pad_image

    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "rus_label": torch.stack([item["rus_label"] for item in batch]),
        "task_label": torch.stack([item["task_label"] for item in batch]),
        "id": [item["id"] for item in batch],
    }

def get_mustard_dataloader(args, tokenizer, image_processor, split):
    if split == "train":
        dataset = MustardDataset(split, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mustard_collate)
    elif split == "val":
        dataset = MustardDataset(split, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=mustard_collate)
    elif split == "test":
        dataset = MustardDataset(split, args.image_data_path, tokenizer, image_processor, args.max_length)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=mustard_collate)
