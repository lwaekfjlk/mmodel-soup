import json
import jsonlines

for dataset_name in ['sarc', 'urfunny', 'mustard']:

    overall_dataset = []

    if dataset_name == 'sarc':
        for i in range(1, 5):
            with jsonlines.open(f'../{dataset_name}_data/data_raw/image_captions_cogvlm2_subpart{i}.jsonl', 'r') as f:
                dataset = list(f)
            overall_dataset.extend(dataset) 
    else:
        with jsonlines.open(f'../{dataset_name}_data/data_raw/image_captions_cogvlm2.jsonl', 'r') as f:
            dataset = list(f)
        overall_dataset.extend(dataset)

    results = {}
    for data in overall_dataset:
        image_id = data['image']
        results[image_id] = data['caption']

    with open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_description_cogvlm2.json', 'w') as f:
        json.dump(results, f, indent=4)
