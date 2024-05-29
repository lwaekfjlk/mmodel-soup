import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def construct_subset(ids, dataset):
    return {id: dataset[id] for id in ids if id in dataset}

def save_dataset(file_path, dataset):
    with open(file_path, 'w') as file:
        json.dump(dataset, file)
