from mustard import MustardDataset
from irfl import IRFLDataset 
from sarc import SarcDataset
from nycartoon import NYCartoonDataset 
from torch.utils.data import Dataset, DataLoader
import torch


class CombinedDataset(Dataset):
    def __init__(self, dataset_configs, tokenizer, image_processor):
        self.datasets = self.load_datasets(dataset_configs, tokenizer, image_processor)

    def load_datasets(self, dataset_configs, tokenizer, image_processor):
        datasets = []
        for config in dataset_configs:
            if config["name"] == "IRFL":
                datasets.append(IRFLDataset(config["dataset_path"], config["image_data_path"], tokenizer, image_processor, config["max_length"]))
            elif config["name"] == "mustard":
                datasets.append(MustardDataset(config["dataset_path"], config["image_data_path"], tokenizer, image_processor, config["max_length"]))
            elif config["name"] == "NYCartoon":
                datasets.append(NYCartoonDataset(config["dataset_path"], config["image_data_path"], tokenizer, image_processor, config["max_length"]))
            elif config["name"] == "sarc":
                datasets.append(SarcDataset(config["dataset_path"], config["image_data_path"], tokenizer, image_processor, config["max_length"]))
        return datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")


def combined_collate(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    id = [item["id"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "image": images,
        "label": labels,
        "id": id,
    }


def get_combined_dataloader(dataset_configs, args, tokenizer, image_processor, split):
    if split == "train":
        dataset = CombinedDataset(dataset_configs, tokenizer, image_processor)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=combined_collate)
    elif split == "val":
        dataset = CombinedDataset(dataset_configs, tokenizer, image_processor)
        return DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=combined_collate)
    elif split == "test":
        dataset = CombinedDataset(dataset_configs, tokenizer, image_processor)
        return DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=combined_collate)

