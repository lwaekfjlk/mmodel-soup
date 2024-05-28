import jsonlines
import json

with jsonlines.open('vision_text_label_logits.jsonl') as reader:
    dataset = list(reader)

formatted_dataset = {}
for data in dataset:
    identifier = data['image_id'].split('.')[0]
    formatted_dataset[identifier] = {}
    formatted_dataset[identifier][data['text']] = {
        'text': data['text'],
        'image_id': data['image_id'],
        'response': data['response'],
        'label': data['label'],
        'figurative_type': data['figurative_type'],
    }
    if len(formatted_dataset[identifier]) > 1:
        print(f"More than one caption for {identifier}")

with open('irfl_pred.json', 'w') as f:
    json.dump(formatted_dataset, f, indent=4)