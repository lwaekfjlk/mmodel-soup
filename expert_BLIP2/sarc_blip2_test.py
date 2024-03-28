import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse
from sarc_dataset import CustomDataset, custom_collate
from sklearn.metrics import f1_score, precision_score, recall_score

def get_test_dataloader(dataset_path, tokenizer, image_processor, batch_size=8, max_length=128):
    """
    Prepare the dataloader for the test dataset.
    """
    dataset = CustomDataset(dataset_path, tokenizer, image_processor, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    return dataloader

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
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    return accuracy, f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model on a test dataset")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for evaluation')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    model = Blip2ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_dataloader = get_test_dataloader(args.test_path, tokenizer, processor, batch_size=args.batch_size)

    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]

    accuracy, f1, precision, recall = evaluate(model, test_dataloader, device, yes_token_id, no_token_id)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
