import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MultiDataset(Dataset):
    def __init__(self, dataset_name, split, image_data_path, tokenizer, image_processor, max_length=512):
        self.dataset_name = dataset_name.lower()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_data_path = image_data_path
        self.max_length = max_length

        if self.dataset_name == "mustard":
            self.dataset_files = self._get_mustard_files(split)
            self.gth = self._load_ground_truth("../mustard_data/data_raw/mustard_dataset_train.json",
                                               "../mustard_data/data_raw/mustard_dataset_test.json")
        elif self.dataset_name == "mmsd":
            self.dataset_files = self._get_mmsd_files(split)
            self.gth = self._load_ground_truth("../mmsd_data/data_raw/mmsd_train.json",
                                               "../mmsd_data/data_raw/mmsd_test.json")
        elif self.dataset_name == "urfunny":
            self.dataset_files = self._get_urfunny_files(split)
            self.gth = self._load_ground_truth("../urfunny_data/data_raw/urfunny_train.json",
                                               "../urfunny_data/data_raw/urfunny_test.json")
        else:
            raise ValueError("Dataset name not recognized. Choose from 'mustard', 'mmsd', or 'urfunny'.")

        self.dataset = self.load_dataset(self.dataset_files)

    def _get_mustard_files(self, split):
        if split == "train":
            return {
                "R": "../mustard_data/data_split_output/mustard_R_dataset_train_cogvlm2_qwen2.json",
                "U": "../mustard_data/data_split_output/mustard_U_dataset_train_cogvlm2_qwen2.json",
                "S": "../mustard_data/data_split_output/mustard_AS_dataset_train_cogvlm2_qwen2.json",
            }
        else:
            return {
                "R": "../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2.json",
                "U": "../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2.json",
                "S": "../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2.json",
            }

    def _get_mmsd_files(self, split):
        if split == "train":
            return {
                "MMSD": "../mmsd_data/data_split_output/mmsd_train.json"
            }
        else:
            return {
                "MMSD": "../mmsd_data/data_split_output/mmsd_test.json"
            }

    def _get_urfunny_files(self, split):
        if split == "train":
            return {
                "URFUNNY": "../urfunny_data/data_split_output/urfunny_train.json"
            }
        else:
            return {
                "URFUNNY": "../urfunny_data/data_split_output/urfunny_test.json"
            }

    def _load_ground_truth(self, train_file, test_file):
        with open(test_file) as f:
            gth = {id: 1 if data["sarcasm"] else 0 for id, data in json.load(f).items()}
        
        with open(train_file) as f:
            gth.update({id: 1 if data["sarcasm"] else 0 for id, data in json.load(f).items()})
        
        return gth

    def load_dataset(self, dataset_files):
        overall_dataset = []
        label_map = {"R": 0, "U": 1, "S": 2, "MMSD": 3, "URFUNNY": 4}

        for type, file_path in dataset_files.items():
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

def get_combined_dataloader(args, tokenizer, image_processor, split):
    datasets = []
    for dataset_name in ["mustard", "mmsd", "urfunny"]:
        dataset = MultiDataset(dataset_name, split, args.image_data_path, tokenizer, image_processor, args.max_length)
        datasets.append(dataset)
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=(split=="train"), collate_fn=mustard_collate)
