import ipdb
import ast
import os
import csv
import json
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, Blip2ForConditionalGeneration, BlipImageProcessor
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = 124
        self.imageProcessor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        image_path = '/root/bak/CogVLM/sarc_data/image_data/' + item[' image']  # Note the space in ' image' key
        image =  Image.open(image_path)
        image = self.imageProcessor(image,
                                        do_resize=True,
                                        size=(128, 128),
                                        return_tensors="pt")
        image = image["pixel_values"][0]
        label = item[' label']

        # Construct prompt
        full_prompt = f"Question: The tweet for this image is: {text}. Is the tweet sarcastic (yes or no)? Answer:"
        
        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),  # Convert label to float
            "tweet": text,
            "prompt": full_prompt
        }

    def tokenize_and_left_pad(self, full_prompt, max_length):
        # Tokenize without padding
        text_encoding = self.tokenizer(full_prompt, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Calculate necessary padding
        seq_len = text_encoding['input_ids'].size(1)
        padding_length = max_length - seq_len
        
        if padding_length > 0:
            # Create padding tensors
            pad_ids = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((1, padding_length), dtype=torch.long)
            
            # Apply left padding
            text_encoding['input_ids'] = torch.cat([pad_ids, text_encoding['input_ids']], dim=1)
            text_encoding['attention_mask'] = torch.cat([pad_mask, text_encoding['attention_mask']], dim=1)
        else:
            # If no padding is necessary, ensure the sequence is truncated to max_length
            text_encoding['input_ids'] = text_encoding['input_ids'][:, :max_length]
            text_encoding['attention_mask'] = text_encoding['attention_mask'][:, :max_length]

        return text_encoding

def custom_collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    tweets = [item["tweet"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["label"] for item in batch]
    images = [item["image"] for item in batch]

    # Pad input_ids and attention_masks
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"input_ids": input_ids, "attention_mask": attention_masks, "image": images, "label": labels, "tweets": tweets, "prompts": prompts }  # Include images in the return dictionary


if __name__ == '__main__':
    # path to data  
    train_path = '/root/bak/mmodel-soup/dataset/sarcasm_blip_test.csv'
    dataset = load_dataset("csv", data_files=train_path)['train']

    # BLIP2 Properties
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("/root/bak/CogVLM/blip_output/model")
    #model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    # prepare data for training
    max_length = 128
    custom_dataset = CustomDataset(dataset, tokenizer)
    train_dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    # train settings
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model.to(device)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    epochs = 1
    total = 0
    correct = 0
    acc = 0
    miss = []
    pred_logits = []
    # Training loop
    for epoch in range(epochs):  # Assuming 3 epochs
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)  # Move image tensor to device
            labels = batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
            if len(outputs.logits.shape) == 2:
                logits = outputs.logits
            else:
                logits = outputs.logits.squeeze() 
            print(outputs.logits.shape)
            logits = outputs.logits[:, -1, :]
            yes_token = tokenizer.tokenize("Yes")
            yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)[0]
            no_token = tokenizer.tokenize("No")
            no_token_id = tokenizer.convert_tokens_to_ids(no_token)[0]
            yes_logits = logits[:, yes_token_id]
            no_logits = logits[:, no_token_id]
            yesno_logits = torch.stack([no_logits, yes_logits], dim=-1)
            pred_logits.append((batch['tweets'], yesno_logits))
            pred_probs = F.softmax(yesno_logits, dim=-1)
            pred_tokens = torch.argmax(pred_probs, dim=-1)
            #print(pred_tokens)
            #print(labels)
            #ipdb.set_trace()
            matches = torch.sum(pred_tokens == labels).item()
            incorrect_indices = (pred_tokens != labels).nonzero(as_tuple=True)[0]
            if len(incorrect_indices) > 0:
                print("Incorrect!! \n")
                miss.append(batch['tweets'][incorrect_indices])
            correct += matches
            total += len(labels)
            # Compute loss
            # Backward pass
            acc = correct/total
            progress_bar.set_postfix(acc=acc)
            # Print loss
        print(f"Accuracy: {acc}")
    print(miss)


pred_logits = [(name, tensor.tolist()) for name, tensor in pred_logits]

with open('pred_logit.json', 'w') as f:
    json.dump(pred_logits, f)
