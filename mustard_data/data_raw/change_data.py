import json

# Read the JSON data from a file
with open('mustard_raw_data_speaker_independent_test.json', 'r') as file:
    data_dict = json.load(file)

# Convert to a list of dictionaries
data_list = [value for key, value in data_dict.items()]

# Write the new list of dictionaries to a new JSON file
with open('test.json', 'w') as file:
    json.dump(data_list, file, indent=4)

print("Data has been written to test.json")



# Read the JSON data from a file
with open('mustard_raw_data_speaker_independent_train.json', 'r') as file:
    data_dict = json.load(file)

# Convert to a list of dictionaries
data_list = [value for key, value in data_dict.items()]

# Write the new list of dictionaries to a new JSON file
with open('train.json', 'w') as file:
    json.dump(data_list, file, indent=4)

print("Data has been written to train.json")