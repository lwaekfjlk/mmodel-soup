import json

def read_and_process_file(dataset, file_path):
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and JSON data
            key, json_data = line.split(' ', 1)
            
            # Parse the JSON data
            parsed_data = json.loads(json_data)
            
            # Store the key and parsed data
            data_list.append((key, parsed_data))
            dataset[key] = parsed_data
    
    return dataset

# Example usage
file_path = './mustard_text_only_pred_test.json'
dataset = {}
dataset = read_and_process_file(dataset, file_path)

formulated_dataset = {}
for id, data in dataset.items():
    logits = {
        'Yes': data['Yes'] if 'Yes' in data else -10000,
        'No': data['No'] if 'No' in data else -10000,
        'yes': data['yes'] if 'yes' in data else -10000,
        'no': data['no'] if 'no' in data else -10000
    }
    yes_logit = logits['Yes']
    no_logit = logits['No']
    if yes_logit > no_logit:
        pred = 1
    elif yes_logit < no_logit:
        pred = 0
    else:
        raise ValueError('Ambiguous prediction')
    formulated_dataset[id] = {
        'logits': logits,
        'pred': pred
    }

with open('./mustard_text_only_pred_test_formulated.json', 'w') as file:
    json.dump(formulated_dataset, file, indent=4)