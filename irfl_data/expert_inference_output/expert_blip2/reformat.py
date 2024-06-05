import json
import jsonlines

for type in ['AS', 'U', 'R', 'baseline']:
    with open('./{}_yesno_logits.json'.format(type), 'r') as f:
        results = json.load(f)

    dataset = []
    for key, value in results.items():
        image_id = key
        logits = value
        dataset.append({'image_id': image_id, 'logits': logits})

    with open('./{}_yesno_logits.jsonl'.format(type), 'w') as f:
        jsonlines.Writer(f).write_all(dataset)