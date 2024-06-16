import json
import jsonlines

id_to_label = {}

# adding label for mustard


with open("../../../nycartoon_data/data_raw/nycartoon_dataset_test.json") as f:
    data = json.load(f)

for key, val in data.items():
    id_to_label[key] = val['label']

for type in ['AS', 'U', 'R', 'baseline']:
    with open('./test_nycartoon_{}_yesno_logits.json'.format(type), 'r') as f:
        results = json.load(f)

    dataset = []
    for key, value in results.items():
        image_id = key.split('_')[0]
        text = key.split('_')[1]
        logits = value
        pred = 1 if logits[0] > logits[1] else 0
        if not (image_id + '_' + text) in id_to_label:
            continue
        target = id_to_label[image_id + '_' + text]
        
        dataset.append({'image_id': image_id, 'text': text, 'logits': logits, 'pred': pred, 'target': target})

    with open('./nycartoon_{}_logits.jsonl'.format(type), 'w') as f:
        jsonlines.Writer(f).write_all(dataset)