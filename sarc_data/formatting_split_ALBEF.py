import json

with open('results/sarc_U_dataset_train.json', 'r') as f:
    data = json.load(f)
    
result = []
for image_id in data:
    result.append([image_id, data[image_id]['text'], data[image_id]['label']])

with open('intermediate_data/ALBEF/sarc_U_dataset_train.txt', 'w') as f:
    for line in result:
        f.write(str(line) + '\n')