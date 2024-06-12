import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from mustard import get_mustard_dataloader
from sarc import get_sarc_dataloader
from nycartoon import get_nycartoon_dataloader
from irfl import get_irfl_dataloader
from combine import get_combined_dataloader
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import ipdb
from peft import LoraConfig, get_peft_model, PeftModel


def evaluate(tokenizer, model, dataloader, device, args):
    model.eval()
    total_correct = 0
    total = 0
    all_labels = []
    total_yesno_logits = {}
    all_predictions = []
    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]
    a_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("1"))[0]
    b_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("2"))[0]
    c_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("3"))[0]
    d_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("4"))[0]
    e_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("5"))[0]
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        ids = batch["id"]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits[:, -1, :]
            if args.answer_options > 2:
                yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            else:
                yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            ipdb.set_trace()
            predictions = torch.argmax(yesno_logits, dim=-1)
            #ipdb.set_trace()
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            yesno_logits = yesno_logits.tolist()
            for i, id in enumerate(ids):
                total_yesno_logits[id] = yesno_logits[i]
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    if args.answer_options == 2:
        accuracy = total_correct / total
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        return accuracy, f1, precision, recall, total_yesno_logits
    else:
        accuracy = total_correct / total
        return accuracy, total_yesno_logits


def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]
    a_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("1"))[0]
    b_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("2"))[0]
    c_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("3"))[0]
    d_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("4"))[0]
    e_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("5"))[0]

    best_f1 = -1
    best_acc = -1
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            if args.answer_options > 2:
                yesno_logits = torch.stack([logits[:, a_token_id], logits[:, b_token_id],  logits[:, c_token_id], logits[:, d_token_id], logits[:, e_token_id]], dim=-1)
            else:
                yesno_logits = torch.stack([logits[:, no_token_id], logits[:, yes_token_id]], dim=-1)
            loss = criterion(yesno_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            #if (step + 1) % args.eval_steps == 0:
        if args.answer_options == 2:
            acc, f1, precision, recall, yesno_logits = evaluate(
                tokenizer, 
                model, 
                test_dataloader, 
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
        else: 
            acc, yesno_logits = evaluate(
                tokenizer, 
                model, 
                test_dataloader, 
                device, 
                args
            )
            print(f"Epoch {epoch + 1} Step {step + 1}")
            print(f"Validation Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                print("SAVING MODEL")
                model.save_pretrained(args.save_path)
                torch.save(yesno_logits, f"{args.save_path}/yesno_logits.pt")

def create_dataset_configs(dataset_names, dataset_paths, image_data_paths, max_lengths):
    configs = []
    for name, path, image_path, max_length in zip(dataset_names, dataset_paths, image_data_paths, max_lengths):
        config = {
            "name": name,
            "dataset_path": path,
            "image_data_path": image_path,
            "max_length": max_length
        }
        configs.append(config)
    return configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the mustard dataset")
    parser.add_argument('--dataset', type=str, default='mustard')
    parser.add_argument('--train_path', type=str, default='../mustard_data/data_split_output/mustard_AS_dataset_train.json', help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='../mustard_data/data_split_output/mustard_dataset_test.json', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='../mustard_data/data_split_output/mustard_dataset_test.json', help='Path to the test data')
    parser.add_argument('--image_data_path', type=str, default='../mustard_data/data_raw/images', help='Path to the image data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length for tokenized sequences')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=10, help='Number of steps between evaluations')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter')
    parser.add_argument('--save_path', type=str, default='./model', help='Path to save the trained model')
    parser.add_argument('--combined_dataset_names', type=str, nargs='+', default=[], help='Names of the datasets to combine')
    parser.add_argument('--combined_train_paths', type=str, nargs='+', default=[], help='Paths to the training data')
    parser.add_argument('--combined_val_paths', type=str, nargs='+', default=[], help='Paths to the validation data')
    parser.add_argument('--combined_test_paths', type=str, nargs='+', default=[], help='Paths to the test data')
    parser.add_argument('--combined_image_data_paths', type=str, nargs='+', default=[], help='Paths to the image data')
    parser.add_argument('--combined_max_lengths', type=int, nargs='+', default=[], help='Maximum lengths for tokenized sequences')
    parser.add_argument('--answer_options', type=int, default=2, help='Maximum number of choices')
    parser.add_argument('--mode', type=str, default="train", help='train/test type')
    parser.add_argument('--load_model_name', type=str, default='./model', help='Path to load the model from')
    parser.add_argument('--device', type=int, default=0, help='specify gpu')
    parser.add_argument('--world_size', type=int, default=4, help='specify gpu')



    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    

    if args.dataset == "mustard":
        train_dataloader = get_mustard_dataloader(args, tokenizer, split="train")
        val_dataloader = get_mustard_dataloader(args, tokenizer, split="val")
        test_dataloader = get_mustard_dataloader(args, tokenizer, split="test")
    elif args.dataset == "sarc":
        train_dataloader = get_sarc_dataloader(args, tokenizer, split="train")
        val_dataloader = get_sarc_dataloader(args, tokenizer, split="val")
        test_dataloader = get_sarc_dataloader(args, tokenizer, split="test")
    elif args.dataset == "nycartoon":
        train_dataloader = get_nycartoon_dataloader(args, tokenizer, split="train")
        val_dataloader = get_nycartoon_dataloader(args, tokenizer, split="val")
        test_dataloader = get_nycartoon_dataloader(args, tokenizer, split="test")
    elif args.dataset == "irfl":
        train_dataloader = get_irfl_dataloader(args, tokenizer, split="train")
        val_dataloader = get_irfl_dataloader(args, tokenizer, split="val")
        test_dataloader = get_irfl_dataloader(args, tokenizer, split="test")
    elif args.dataset == "combined":
        train_configs = create_dataset_configs(args.combined_dataset_names, args.combined_train_paths, args.combined_image_data_paths, args.combined_max_lengths)
        val_configs = create_dataset_configs(args.combined_dataset_names, args.combined_val_paths, args.combined_image_data_paths, args.combined_max_lengths)
        test_configs = create_dataset_configs(args.combined_dataset_names, args.combined_test_paths, args.combined_image_data_paths, args.combined_max_lengths)
        train_dataloader = get_combined_dataloader(train_configs, args, tokenizer, split="train")
        val_dataloader = get_combined_dataloader(val_configs, args, tokenizer, split="val")
        test_dataloader = get_combined_dataloader(test_configs, args, tokenizer, split="test")
    
    if args.mode == "train":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
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
        train(model, train_dataloader, val_dataloader, tokenizer, device, args)

        if args.answer_options == 2:
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
        else: 
            acc, yesno_logits = evaluate(
                tokenizer, 
                model, 
                test_dataloader, 
                device, 
                args
            )
            print("Test Results:")
            print(f"Test Accuracy: {acc:.4f}")


#        model.save_pretrained(args.save_path)
    else:
        print(f"TEST {args.load_model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.load_model_name).to(device)
        #model = PeftModel.from_pretrained(
        #    model,
        #    args.load_model_name,
        #    is_trainable=True
        #).to(device)
        if args.answer_options == 2:
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
            with open(f"./{args.load_model_name}/test_yesno_logits.json", "w") as f:
                json.dump(yesno_logits, f)
            print(acc, f1, precision, recall)
        else: 
            acc, yesno_logits = evaluate(
                tokenizer, 
                model, 
                test_dataloader, 
                device, 
                args
            )
            print("Test Results:")
            print(f"Test Accuracy: {acc:.4f}")
            with open(f"./{args.load_model_name}/test_yesno_logits.json", "w") as f:
                json.dump(yesno_logits, f)
