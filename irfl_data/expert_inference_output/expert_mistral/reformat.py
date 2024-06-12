import json
import ipdb
import jsonlines

id_to_label = {}

with open("/storage/mmodel-soup/irfl_data/data_raw/irfl_dataset_test.json") as f:
    data = json.load(f)

for key, val in data.items():
    id_to_label[key] = 1 if 'Figurative' in val ['category'] else 0


for type in ['AS', 'U', 'R', 'baseline']:
    with open('./irfl_{}_yesno_logits.json'.format(type), 'r') as f:
        results = json.load(f)

    dataset = []
    for key, value in results.items():
        image_id = key
        logits = value
        pred = 1 if logits[0] > logits[1] else 0
        if not image_id in id_to_label:
            continue
        target = id_to_label[image_id]
        dataset.append({'image_id': image_id, 'logits': logits, 'pred': pred, 'target': target})

    with open('./irfl_{}_logits.jsonl'.format(type), 'w') as f:
        jsonlines.Writer(f).write_all(dataset)