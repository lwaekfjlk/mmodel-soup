import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import argparse

class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, image_processor, max_length=128):
        self.dataset = load_dataset("csv", data_files=dataset_path)['train']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        image_path = f'/root/bak/CogVLM/sarc_data/image_data/{item[" image"].strip()}'
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        label = torch.tensor(item[' label'], dtype=torch.long)

        full_prompt = f"Question: The tweet for this image is: {text}. Is the tweet sarcastic (yes or no)? Answer:"
        text_encoding = self.tokenizer(full_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        return { 
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
            "tweet": text,
            "prompt": full_prompt
        }

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

def evaluate(model, dataloader, device, yes_token_id, no_token_id):
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
            yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            predictions = torch.argmax(yesno_logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = total_correct / total
    return accuracy

def train(args, model, train_dataloader, val_dataloader, tokenizer, device):
    """
    Training loop for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Precompute token IDs for "yes" and "no" responses
    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]

    step = 0
    best_acc = -1
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits.squeeze()[:, -1, :]
            yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            loss = criterion(yesno_logits, labels)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                acc = evaluate(model, val_dataloader, device, yes_token_id, no_token_id)
                print(f"Accuracy: {acc}")
                if best_acc < acc:
                    best_acc = acc
                    model.save_pretrained(args.save_path)

            total_loss += loss.item()

def get_dataloader(dataset_path, tokenizer, image_processor, batch_size=8, max_length=128):
    """
    Get the dataloader for the given dataset.
    """
    custom_dataset = CustomDataset(dataset_path, tokenizer, image_processor, max_length)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on sarcastic image-text pairs")

    # Define arguments
    parser.add_argument('--train_path', type=str, default='./sarc_blip2_train.csv', help='Path to training dataset')
    parser.add_argument('--val_path', type=str, default='./sarc_blip2_test.csv', help='Path to validation dataset')
    parser.add_argument('--train_batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=48, help='Batch size for validation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Lora alpha value')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Lora dropout value')
    parser.add_argument('--lora_rank', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--save_path', type=str, default='./sarc_blip2_model', help='Path to save the trained model')
    parser.add_argument('--eval_step', type=int, default=100, help='Number of steps to evaluate the model')
    args = parser.parse_args()

    # BLIP2 Properties
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)


    train_dataloader = get_dataloader(
        args.train_path, 
        tokenizer, 
        processor, 
        batch_size=args.train_batch_size, 
        max_length=args.max_length
    )
    val_dataloader = get_dataloader(
        args.val_path, 
        tokenizer, 
        processor, 
        batch_size=args.val_batch_size, 
        max_length=args.max_length
    )

    train(
        args,
        model, 
        train_dataloader, 
        val_dataloader, 
        tokenizer, 
        device, 
    )
    model.save_pretrained(args.save_path)
