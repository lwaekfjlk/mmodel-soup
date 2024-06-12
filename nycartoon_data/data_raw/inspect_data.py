import json
import ipdb

file_path = 'nycartoon_dataset_train.json'

with open(file_path, 'r') as file:
    data = json.load(file)

for key in data:
    data[key]['mode'] = 'single'


with open(f"nycartoon_dataset_train_NEW.json", 'w') as f:
    json.dump(data, f, indent=4)
