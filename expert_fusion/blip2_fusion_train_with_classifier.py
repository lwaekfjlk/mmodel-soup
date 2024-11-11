import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import json

from mmsd_with_classifier import get_mmsd_dataloader
from urfunny import get_urfunny_dataloader
from mustard_with_classifier import get_mustard_dataloader
import pickle
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class Blip2WithClassifier(nn.Module):
    def __init__(self, config):
        super(Blip2WithClassifier, self).__init__()
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.config = config
        self.blip2 = get_peft_model(model, self.config)
        self.rus_classifier = nn.Linear(self.blip2.config.text_config.hidden_size, 3)
        self.task_classifier = nn.Linear(self.blip2.config.text_config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.blip2(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        hidden_states = outputs.language_model_outputs.hidden_states[-1]  # Get the last hidden state

        # Use the hidden states to compute logits for classification
        rus_logits = self.rus_classifier(hidden_states[:, -1, :])
        task_logits = self.task_classifier(hidden_states[:, -1, :])

        return rus_logits, task_logits

    def save(self, path):
        # Save the BLIP-2 model
        self.blip2.save_pretrained(path)
        
        # Save the classifiers' state dicts
        torch.save(self.rus_classifier.state_dict(), f"{path}/rus_classifier.pth")
        torch.save(self.task_classifier.state_dict(), f"{path}/task_classifier.pth")

    def load(self, path):
        # Load the BLIP-2 model
        blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip2 = PeftModel.from_pretrained(blip2, path, is_trainable=True)
        
        # Load the classifiers' state dicts

        # path has rus and task classifier state dicts
        if f"rus_classifier.pth" in os.listdir(path):
            print('loading rus classifier')
            self.rus_classifier.load_state_dict(torch.load(f"{path}/rus_classifier.pth"))
        else:
            self.rus_classifier = nn.Linear(self.blip2.config.text_config.hidden_size, 3)
        
        if f"task_classifier.pth" in os.listdir(path):
            print('loading task classifier')
            self.task_classifier.load_state_dict(torch.load(f"{path}/task_classifier.pth"))
        else:
            self.task_classifier = nn.Linear(self.blip2.config.text_config.hidden_size, 2)

        return self
        


def evaluate(tokenizer, model, dataloader, device):
    model.eval()
    total_rus_correct, total_task_correct, total = 0, 0, 0
    all_rus_labels, all_rus_preds = [], []
    all_task_labels, all_task_preds = [], []
    rus_logits_dict = {}
    criterion = nn.CrossEntropyLoss()

    val_loss = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids, attention_mask, images, rus_labels, task_labels = (
            batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "rus_label", "task_label"]
        )
        ids = batch["id"]

        with torch.no_grad():
            rus_logits, task_logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)

            rus_probs = torch.softmax(rus_logits, dim=-1)
            rus_preds = torch.argmax(rus_logits, dim=-1)
            task_preds = torch.argmax(task_logits, dim=-1)

            # Update total correct predictions
            total_rus_correct += (rus_preds == rus_labels).sum().item()
            total_task_correct += (task_preds == task_labels).sum().item()
            total += rus_labels.size(0)  # Assuming rus_labels and task_labels are of the same size

            val_loss += criterion(rus_logits, rus_labels)

            # Store logits and predictions for metrics calculation
            for i, id in enumerate(ids):
                rus_logits_dict[id] = rus_logits[i].tolist()
            
            all_rus_labels.extend(rus_labels.cpu().tolist())
            all_rus_preds.extend(rus_preds.cpu().tolist())
            all_task_labels.extend(task_labels.cpu().tolist())
            all_task_preds.extend(task_preds.cpu().tolist())

    total_val_loss = val_loss / len(dataloader)

    rus_metrics = {
        "accuracy": total_rus_correct / total,
        "f1": f1_score(all_rus_labels, all_rus_preds, average="macro"),
        "precision": precision_score(all_rus_labels, all_rus_preds, average=None),
        "recall": recall_score(all_rus_labels, all_rus_preds, average=None),
    }

    task_metrics = {
        "accuracy": total_task_correct / total,
        "f1": f1_score(all_task_labels, all_task_preds),
        "precision": precision_score(all_task_labels, all_task_preds, average=None),
        "recall": recall_score(all_task_labels, all_task_preds),
    }

    return {
        "rus_metrics": rus_metrics,
        "task_metrics": task_metrics,
        "rus_logits": rus_logits_dict,
        "val_loss": total_val_loss
    }


def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rus_criterion = FocalLoss(alpha=0.25, gamma=2)
    task_criterion = FocalLoss(alpha=0.25, gamma=2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.epochs)

    best_f1 = -1
    best_val_loss = 1000000
    total_step = 0
    rus_losses = []
    task_losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            total_step += 1
            input_ids, attention_mask, images, rus_labels, task_labels = (
                batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "rus_label", "task_label"]
            )

            optimizer.zero_grad()
            rus_logits, task_logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=images
            )


            # Compute losses for both tasks
            rus_loss = rus_criterion(rus_logits, rus_labels)
            task_loss = task_criterion(task_logits, task_labels)
            rus_losses.append(rus_loss)
            task_losses.append(task_loss)
            #loss = rus_loss + task_loss
            loss = task_loss
            print('rus_loss: ', rus_loss)
            print('task_loss: ', task_loss)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if (total_step + 1) % args.eval_steps == 0:
                pickle.dump(rus_losses, open('rus_losses_mmsd_without_task_loss.pkl', 'wb'))
                pickle.dump(task_losses, open('task_losses_mmsd_without_task_loss.pkl', 'wb'))
                metrics = evaluate(tokenizer, model, val_dataloader, device)
                rus_metrics = metrics['rus_metrics']
                task_metrics = metrics['task_metrics']
                val_loss = metrics['val_loss']


                print(f"Epoch {epoch + 1} Step {step + 1} - "
                      f"RUS Val loss: {val_loss:.4f}, "
                      f"RUS Val F1: {rus_metrics['f1']:.4f}, "
                )

                print(f"RUS metric: {rus_metrics['f1']}, {rus_metrics['precision']}, {rus_metrics['recall']}")
                print(f"Task metric: {task_metrics['f1']}, {task_metrics['precision']}, {task_metrics['recall']}")

                if val_loss < best_val_loss:
                    print(f"Saving model with best val loss: {val_loss:.4f}")
                    best_val_loss = val_loss
                    model.save(f"{args.save_path}_best_val_loss")
                    with open(f"{args.save_path}_best_val_loss/rus_logits.json", "w") as f:
                        json.dump(metrics["rus_logits"], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the dataset")
    parser.add_argument('--mode', type=str, default='train', help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--image_data_path', type=str, required=True, help='Path to the image data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=40, help='Batch size for validation')
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
        config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            bias="none", 
            target_modules=["q_proj", "k_proj"]
        )
        model = Blip2WithClassifier(config)
        if args.load_from_ckpt:
            model = model.load(args.load_from_ckpt)
        else:
            model = get_peft_model(model, config)

        #model.print_trainable_parameters()
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
        config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            bias="none", 
            target_modules=["q_proj", "k_proj"]
        )
        model = Blip2WithClassifier(config)
        model = model.load(args.load_model_name)
        model.to(device)

        dataloaders = {
            "mmsd": get_mmsd_dataloader,
            "urfunny": get_urfunny_dataloader,
            "mustard": get_mustard_dataloader
        }

        test_dataloader = dataloaders[args.dataset](args, tokenizer, processor, split="test")
        metrics = evaluate(tokenizer, model, test_dataloader, device)
        
        with open(f"./{args.load_model_name}/test_rus_logits.json", "w") as f:
            json.dump(metrics["rus_logits"], f)

        print(metrics)        
