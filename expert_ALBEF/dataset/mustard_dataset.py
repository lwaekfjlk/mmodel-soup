import json
import os

from dataset.utils import pre_caption
from PIL import Image
from torch.utils.data import Dataset


class mustard_train_dataset(Dataset):
    def __init__(self, dataset_path, transform, image_root, max_words=150):
        self.dataset = self.load_dataset(dataset_path)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        result = []
        for id, data in raw_dataset.items():
            image_id = id
            text = data["utterance"]
            label = 1 if data["sarcasm"] == True else 0
            result.append({"image_id": image_id, "text": text, "label": label})
        return result

    def __getitem__(self, index):
        image_id, text, label = (
            self.dataset[index]["image_id"],
            self.dataset[index]["text"],
            self.dataset[index]["label"],
        )
        image_path = os.path.join(self.image_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = pre_caption(text, self.max_words)

        return image, text, label


class mustard_test_dataset(Dataset):
    def __init__(self, dataset_path, transform, image_root, max_words=150):
        self.dataset = self.load_dataset(dataset_path)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, dataset_path):
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        result = []
        for id, data in raw_dataset.items():
            image_id = id
            text = data["utterance"]
            label = 1 if data["sarcasm"] == True else 0
            result.append({"image_id": image_id, "text": text, "label": label})
        return result

    def __getitem__(self, index):
        image_id, text, label = (
            self.dataset[index]["image_id"],
            self.dataset[index]["text"],
            self.dataset[index]["label"],
        )
        image_path = os.path.join(self.image_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = pre_caption(text, self.max_words)

        return image, text, label, image_id
