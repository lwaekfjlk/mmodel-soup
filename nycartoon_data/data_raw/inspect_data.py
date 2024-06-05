import json
import ipdb

file_path = 'nycartoon_dataset_test.json'

with open(file_path, 'r') as file:
    data = json.load(file)

for key in data:
    split_key = key.split('_')
    if len(split_key) != 2:
        print(len(split_key))

ipdb.set_trace()