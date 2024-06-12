import os
import json
import ipdb

# Define the directory where the JSON files are located
train_path = '/storage/mmodel-soup/nycartoon_data/data_raw/nycartoon_dataset_test.json'
with open(train_path, 'r') as file:
    # Load the JSON data
    test_data = json.load(file)

for observation in test_data:
    test_data[observation]['mode'] = "multichoice"

file_path = '/storage/mmodel-soup/nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json'
with open(f"{file_path}_NEW.json", "w") as file:
    # Write the dictionary to the file in JSON format
    json.dump(test_data, file, indent=4)