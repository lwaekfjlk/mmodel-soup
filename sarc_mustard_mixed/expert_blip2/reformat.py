import json
import jsonlines

id_to_label = {}

# adding label for mustard

with open("../../sarc_data/data_raw/sarc_dataset_test.json") as f:
    data = json.load(f)

for key, val in data.items():
    id_to_label[key] = val['label']

for type in ['AS', 'U', 'R', 'baseline']:
    with open('./sarc_mustard_{}_yesno_logits.json'.format(type), 'r') as f:
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

    with open('./sarc_{}_logits.jsonl'.format(type), 'w') as f:
        jsonlines.Writer(f).write_all(dataset)