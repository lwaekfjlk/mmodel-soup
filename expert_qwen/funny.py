import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ipdb


class FunnyDataset(Dataset):
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
                "context": data["context_sentences"],
                "utterance": data["punchline_sentence"],
                "label": data["label"],
            } 
            for id, data in raw_dataset.items()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['utterance']
        id = item['id']
        context = item['context']
        input_json_file = "/storage/mmodel-soup/funny_data/data_gen_output/image_captions_output.json"
        # Read the dictionary from the JSON file
        with open(input_json_file, "r") as json_file:
            image_description_dict = json.load(json_file)
        caption = image_description_dict[id]
        fullContext = "The dialogue so far has been:"
        for i in range(len(context)):
            currentUtterance = f"{context[i]} \n"
            fullContext += currentUtterance
        label = torch.tensor(item['label'], dtype=torch.long)
        full_prompt = f"Predict if the punchline given an image captioned as {caption} is humorous. {fullContext}. The punchline is {text}. Following up to this image input, the dialogue has been {fullContext}. Is the punchline humorous? Answer:"
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


def funny_collate(batch):
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


def get_funny_dataloader(args, tokenizer,  split):
    if split == "train":
        dataset = FunnyDataset(args.train_path, tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=funny_collate)
    elif split == "val":
        dataset = FunnyDataset(args.val_path, tokenizer,  args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=funny_collate)
    elif split == "test":
        dataset = FunnyDataset(args.test_path,  tokenizer,args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=funny_collate)
