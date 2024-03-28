import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#taken from 
def get_image_path_from_hugginface_cache(image_name):
    chached_image_path = IRFL_images[0]['image'].filename
    chached_image_name = chached_image_path.split('/')[-1]
    return chached_image_path.replace(chached_image_name, image_name.split('.')[0] + '.jpeg')

def get_image(image_name):
  image_path = get_image_path_from_hugginface_cache(image_name)
  return Image.open(image_path)

class CustomDataset(Dataset):
    """
    A custom dataset class that prepares image-text pairs for training.
    """
    def __init__(self, dataset_path, tokenizer, image_processor, max_length=512):
        self.dataset = load_dataset("json", data_files=dataset_path)['train']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        image_id = item['image_id']
        image = get_image(image_id)
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        category = item['type']
        label = torch.tensor(item['label'], dtype=torch.long)
        full_prompt = f"Question: You are given an image text pair of a {category}. The text is: {text}. Can you answer 1 if the {category} is Partial Literal (Some objects/entities of the phrase are visualized) and 0 if the {category} is Figurative (The image conveys one or more definitions of the phrase)? Answer:"
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


def get_test_dataloader(dataset_path, tokenizer, image_processor, batch_size=8, max_length=512):
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
    all_labels = []
    all_predictions = []
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
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    accuracy = total_correct / total
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    return accuracy, f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model on a test dataset")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for evaluation')
    args = parser.parse_args()
    IRFL_images = load_dataset("lampent/IRFL", data_files='IRFL_images.zip')['train']
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    model = Blip2ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_dataloader = get_test_dataloader(args.test_path, tokenizer, processor, batch_size=args.batch_size)

    yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]

    accuracy, f1, precision, recall = evaluate(model, test_dataloader, device, yes_token_id, no_token_id)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")