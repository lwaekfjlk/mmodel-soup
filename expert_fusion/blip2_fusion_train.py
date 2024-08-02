import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import json

from mmsd import get_mmsd_dataloader
from urfunny import get_urfunny_dataloader
from mustard import get_mustard_dataloader


def evaluate(tokenizer, model, dataloader, device):
    model.eval()
    total_correct, total = 0, 0
    all_labels, all_predictions = [], []
    rus_logits = {}
    token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in ["R", "U", "S"]}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids, attention_mask, images, labels = (batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "label"])
        ids = batch["id"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images).logits[:, -1, :]
            rus_logits_batch = torch.stack([logits[:, token_ids[token]] for token in ["R", "U", "S"]], dim=-1)
            predictions = torch.argmax(rus_logits_batch, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for i, id in enumerate(ids):
                rus_logits[id] = rus_logits_batch[i].tolist()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return {
        "accuracy": total_correct / total,
        "f1": f1_score(all_labels, all_predictions, average="macro"),
        "precision": precision_score(all_labels, all_predictions, average="macro"),
        "recall": recall_score(all_labels, all_predictions, average="macro"),
        "rus_logits": rus_logits
    }


def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in ["R", "U", "S"]}
    best_f1 = -1

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            input_ids, attention_mask, images, labels = (batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "label"])

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images).logits[:, -1, :]
            rus_logits_batch = torch.stack([logits[:, token_ids[token]] for token in ["R", "U", "S"]], dim=-1)
            loss = criterion(rus_logits_batch, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % args.eval_steps == 0:
                metrics = evaluate(tokenizer, model, val_dataloader, device)
                print(f"Epoch {epoch + 1} Step {step + 1} - Val Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    model.save_pretrained(args.save_path)
                    with open(f"{args.save_path}/rus_logits.json", "w") as f:
                        json.dump(metrics["rus_logits"], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the dataset")
    parser.add_argument('--mode', type=str, default='train', help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--image_data_path', type=str, required=True, help='Path to the image data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length for tokenized sequences')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=10, help='Number of steps between evaluations')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter')
    parser.add_argument('--save_path', type=str, default='./model', help='Path to save the trained model')
    parser.add_argument('--load_model_name', type=str, help='Path to load the model from')
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='Path to load the model from')
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        if args.load_from_ckpt:
            model = PeftModel.from_pretrained(model, args.load_from_ckpt, is_trainable=True)
        else:
            config = LoraConfig(
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                bias="none", 
                target_modules=["q_proj", "k_proj"]
            )
            model = get_peft_model(model, config)

        model.print_trainable_parameters()
        model.to(device)

        dataloaders = {
            "mmsd": get_mmsd_dataloader,
            "urfunny": get_urfunny_dataloader,
            "mustard": get_mustard_dataloader,
        }

        train_dataloader = dataloaders[args.dataset](args, tokenizer, processor, split="train")
        val_dataloader = dataloaders[args.dataset](args, tokenizer, processor, split="val")

        train(model, train_dataloader, val_dataloader, tokenizer, device, args)

        test_dataloader = dataloaders[args.dataset](args, tokenizer, processor, split="test")
        metrics = evaluate(tokenizer, model, test_dataloader, device)
        print(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    elif args.mode == "test":
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = PeftModel.from_pretrained(model, args.load_model_name, is_trainable=True).to(device)

        dataloaders = {
            "mmsd": get_mmsd_dataloader,
            "urfunny": get_urfunny_dataloader,
            "mustard": get_mustard_dataloader
        }

        test_dataloader = dataloaders[args.dataset](args, tokenizer, processor, split="test")
        metrics = evaluate(tokenizer, model, test_dataloader, device)
        
        with open(f"./{args.load_model_name}/test_rus_logits.json", "w") as f:
            json.dump(metrics["rus_logits"], f)
        
        print(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
