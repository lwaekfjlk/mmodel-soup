import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from mustard import get_mustard_dataloader
from sarc import get_sarc_dataloader
from nycartoon import get_nycartoon_dataloader
from irfl import get_irfl_dataloader
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(tokenizer, model, dataloader, device, args):
    model.eval()
    total_correct = 0
    total = 0
    total_yesno_logits = []
    all_labels = []
    all_predictions = []
    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits
            logits = logits[:, -1, :]

            yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            predictions = torch.argmax(yesno_logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_yesno_logits.extend(yesno_logits.tolist())
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    
    accuracy = total_correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    
    return accuracy, f1, precision, recall, total_yesno_logits


def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]

    best_f1 = -1
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits[:, -1, :]
            yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            loss = criterion(yesno_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % args.eval_steps == 0:
                acc, f1, precision, recall, yesno_logits = evaluate(
                    tokenizer, 
                    model, 
                    val_dataloader, 
                    device, 
                    args
                )
                print(f"Epoch {epoch + 1} Step {step + 1}")
                print(f"Validation Accuracy: {acc:.4f}")
                print(f"Validation F1 Score: {f1:.4f}")
                print(f"Validation Precision: {precision:.4f}")
                print(f"Validation Recall: {recall:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    model.save_pretrained(args.save_path)
                    torch.save(yesno_logits, f"{args.save_path}/yesno_logits.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the mustard dataset")
    parser.add_argument('--dataset', type=str, default='mustard')
    parser.add_argument('--train_path', type=str, default='../../mustard_data/data_split_output/mustard_AS_dataset_train.json', help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='../../mustard_data/data_split_output/mustard_dataset_test.json', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='../../mustard_data/data_split_output/mustard_dataset_test.json', help='Path to the test data')
    parser.add_argument('--image_data_path', type=str, default='../../mustard_data/data_raw/images', help='Path to the image data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length for tokenized sequences')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=10, help='Number of steps between evaluations')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter')
    parser.add_argument('--save_path', type=str, default='./model', help='Path to save the trained model')
    
    args = parser.parse_args()

    # BLIP2 Properties
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
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

    if args.dataset == "mustard":
        train_dataloader = get_mustard_dataloader(args, tokenizer, processor, split="train")
        val_dataloader = get_mustard_dataloader(args, tokenizer, processor, split="val")
    elif args.dataset == "sarc":
        train_dataloader = get_sarc_dataloader(args, tokenizer, processor, split="train")
        val_dataloader = get_sarc_dataloader(args, tokenizer, processor, split="val")
    elif args.dataset == "nycartoon":
        train_dataloader = get_nycartoon_dataloader(args, tokenizer, processor, split="train")
        val_dataloader = get_nycartoon_dataloader(args, tokenizer, processor, split="val")
    elif args.dataset == "irfl":
        train_dataloader = get_irfl_dataloader(args, tokenizer, processor, split="train")
        val_dataloader = get_irfl_dataloader(args, tokenizer, processor, split="val")

    train(model, train_dataloader, val_dataloader, tokenizer, device, args)

    # Test the model
    if args.dataset == 'mustard':
        test_dataloader = get_mustard_dataloader(args, tokenizer, processor, split="test")
    elif args.dataset == 'sarc':
        test_dataloader = get_sarc_dataloader(args, tokenizer, processor, split="test")
    elif args.dataset == 'nycartoon':
        test_dataloader = get_nycartoon_dataloader(args, tokenizer, processor, split="test")
    elif args.dataset == 'irfl':
        test_dataloader = get_irfl_dataloader(args, tokenizer, processor, split="test")

    acc, f1, precision, recall, yesno_logits = evaluate(
        tokenizer, 
        model, 
        test_dataloader, 
        device, 
        args
    )
    print("Test Results:")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    model.save_pretrained(args.save_path)
