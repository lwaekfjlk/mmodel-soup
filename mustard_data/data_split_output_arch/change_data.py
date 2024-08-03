import json

# Read the JSON data from a file
with open('mustard_R_dataset_train.json', 'r') as file:
    data_dict = json.load(file)

# Convert to a list of dictionaries
data_list = [value for key, value in data_dict.items()]

# Write the new list of dictionaries to a new JSON file
with open('mustard_R_dataset_train_new.json', 'w') as file:
    json.dump(data_list, file, indent=4)


# Read the JSON data from a file
with open('mustard_AS_dataset_train.json', 'r') as file:
    data_dict = json.load(file)

# Convert to a list of dictionaries
data_list = [value for key, value in data_dict.items()]

# Write the new list of dictionaries to a new JSON file
with open('mustard_AS_dataset_train_new.json', 'w') as file:
    json.dump(data_list, file, indent=4)


# Read the JSON data from a file
with open('mustard_U_dataset_train.json', 'r') as file:
    data_dict = json.load(file)

# Convert to a list of dictionaries
data_list = [value for key, value in data_dict.items()]

# Write the new list of dictionaries to a new JSON file
with open('mustard_U_dataset_train_new.json', 'w') as file:
    json.dump(data_list, file, indent=4)


