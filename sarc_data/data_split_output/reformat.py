import json
import ast


def process_data(data):
    processed_data = []
    for entry in data:
        # Convert the string representation of the list into an actual list
        entry_list = ast.literal_eval(entry)
        processed_data.append(entry_list)
    return processed_data

splits = ['AS', 'R', 'U']

for split in splits:

    formatted_dataset = {}
    with open('./sarc_{}_dataset_train.txt'.format(split)) as f:
        dataset = f.readlines()
    dataset = process_data(dataset)


    for data in dataset:
        image_id = data[0]
        text = data[1]
        label = data[2]
        formatted_dataset[image_id] = {
            "iamge_id": image_id,
            "text": text,
            "label": label
        }

    with open('./sarc_{}_dataset_train_history_version.json'.format(split), 'w') as f:
        json.dump(formatted_dataset, f, indent=4)