import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import ipdb
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime


class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, image_processor, split, max_length=512):
        self.dataset = load_dataset(dataset_path, "matching")[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        choices = item['caption_choices']
        caption_a, caption_b, caption_c, caption_d, caption_e  = choices[0], choices[1],choices[2],choices[3], choices[4]
        image = item['image']
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labelTranslate = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        label = labelTranslate[item['label']]
    
        label = torch.tensor(label, dtype=torch.long)
        full_prompt = f"Question: You are given an image and 5 caption choices. 1: {caption_a}. 2: {caption_b}. 3: {caption_c}. 4: {caption_d}. 5: {caption_e}. Can you return which choice suits the image best? Answer:"
        # right padding
        #ipdb.set_trace()s
        #text_encoding = self.tokenize(full_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        # left padding
        text_encoding = self.tokenize_and_left_pad(full_prompt, self.max_length)
        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
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
    """
    A custom collate function to pad the batches dynamically.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "image": images,
        "label": labels
    }

def evaluate(model, dataloader, device, a_token_id, b_token_id, c_token_id, d_token_id, e_token_id):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    total_correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
            logits = outputs.logits.squeeze()[:, -1, :]
            yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            predictions = torch.argmax(yesno_logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = total_correct / total
    return accuracy


def train(model, train_dataloader, val_dataloader, tokenizer, device, epochs=5):
    """
    Training loop for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Precompute token IDs for "yes" and "no" responses
    a_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("1"))[0]
    b_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("2"))[0]
    c_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("3"))[0]
    d_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("4"))[0]
    e_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("5"))[0]
    acc = evaluate(model, val_dataloader, device, a_token_id, b_token_id, c_token_id, d_token_id, e_token_id)
    print("ACCURACY BASELINE", acc)
    step = 0
    best_acc = -1
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            #ipdb.set_trace()
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits.squeeze()[:, -1, :]
            yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            loss = criterion(yesno_logits, labels)
            print("LOSS:", loss)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                acc = evaluate(model, val_dataloader, device, a_token_id, b_token_id, c_token_id, d_token_id, e_token_id)
                print(f"Test Accuracy: {acc}")
                if best_acc < acc:
                    best_acc = acc
                    model.save_pretrained("./model")

            total_loss += loss.item()

def get_dataloader(dataset_path, tokenizer, image_processor, split, batch_size=8, max_length=128):
    """
    Get the dataloader for the given dataset.
    """
    custom_dataset = CustomDataset(dataset_path, tokenizer, image_processor, split, max_length)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    return dataloader


if __name__ == '__main__':
    # path to data  
    # BLIP2 Properties
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = "cuda:6" if torch.cuda.is_available() else "cpu" 
    model = Blip2ForConditionalGeneration.from_pretrained("./modelMatchingFineTune")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    train_path = "jmhessel/newyorker_caption_contest"
    val_path = "jmhessel/newyorker_caption_contest"
    train_dataloader = get_dataloader(train_path, tokenizer, processor, split = "train", batch_size=24, max_length=128)
    val_dataloader = get_dataloader(val_path, tokenizer, processor, split = "test", batch_size=32, max_length=128)
    train(model, train_dataloader, val_dataloader, tokenizer, device, epochs=5)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"./modelMatchingFineTune_{timestamp}"
    model.save_pretrained("./modelMatchingFineTune")

