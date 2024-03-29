import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import argparse
from nyt_dataset_matching import CustomDataset, custom_collate
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
import ipdb

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
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            predictions = torch.argmax(yesno_logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = total_correct / total
    return accuracy

def train(args, model, train_dataloader, val_dataloader, tokenizer, device, epochs=50):
    """
    Training loop for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Precompute token IDs for "yes" and "no" responses
    a_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("1"))[0]
    b_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("2"))[0]
    c_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("3"))[0]
    d_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("4"))[0]
    e_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("5"))[0]
    #acc = evaluate(model, val_dataloader, device, a_token_id, b_token_id, c_token_id, d_token_id, e_token_id)
    #print("ACCURACY BASELINE", acc)
    step = 0
    best_acc = -1
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            #ipdb.set_trace()
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            loss = criterion(yesno_logits, labels)
            print("LOSS:", loss)
            loss.backward()
            optimizer.step()
            step += 1
            """             if step % 50 == 0:
                acc = evaluate(model, val_dataloader, device, a_token_id, b_token_id, c_token_id, d_token_id, e_token_id)
                print(f"Test Accuracy: {acc}")
                if best_acc < acc:
                    best_acc = acc
                    model.save_pretrained("./model")
            """

            total_loss += loss.item()

def get_dataloader(dataset_path, tokenizer, split, batch_size=8, max_length=512):
    """
    Get the dataloader for the given dataset.
    """
    custom_dataset = CustomDataset(dataset_path, tokenizer, split, max_length)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on sarcastic image-text pairs")

    # Define arguments
    parser.add_argument('--train_path', type=str, default="jmhessel/newyorker_caption_contest", help='Path to training dataset')
    parser.add_argument('--val_path', type=str, default="jmhessel/newyorker_caption_contest", help='Path to validation dataset')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Lora alpha value')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Lora dropout value')
    parser.add_argument('--lora_rank', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--save_path', type=str, default='./nytMatching_mistral_model', help='Path to save the trained model')
    parser.add_argument('--eval_step', type=int, default=100, help='Number of steps to evaluate the model')
    args = parser.parse_args()

    # mistral Properties
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    device = "cuda:5" if torch.cuda.is_available() else "cpu" 
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
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
        "train",
        batch_size=args.train_batch_size, 
        max_length=args.max_length
    )
    val_dataloader = get_dataloader(
        args.val_path, 
        tokenizer, 
        "validation",
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
