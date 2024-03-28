import json

with open('../mustard_data/results/mustard_image_description.json', 'r') as f:
    mustard_image_description = json.load(f)

with open('../mustard_data/intermediate_data/sarcasm_data_speaker_independent_train.json', 'r') as f:
    mustard_train_data = json.load(f)

with open('../mustard_data/intermediate_data/sarcasm_data_speaker_independent_test.json', 'r') as f:
    mustard_test_data = json.load(f)

image_text_train_dataset = {}
for inst_id in mustard_train_data:
    inst = mustard_train_data[inst_id]
    image_description = mustard_image_description[inst_id]
    utterance = inst['utterance']
    label = inst['sarcasm']
    image_text_pair_data = {
        'vision': image_description,
        'text': utterance,
        'label': label,
    }
    image_text_train_dataset[inst_id] = image_text_pair_data

with open('../mustard_data/results/image_text_train_dataset.json', 'w') as f:
    json.dump(image_text_train_dataset, f)

image_text_test_dataset = {}
for inst_id in mustard_test_data:
    inst = mustard_test_data[inst_id]
    image_description = mustard_image_description[inst_id]
    utterance = inst['utterance']
    label = inst['sarcasm']
    image_text_pair_data = {
        'vision': image_description,
        'text': utterance,
        'label': label,
    }
    image_text_test_dataset[inst_id] = image_text_pair_data

with open('../mustard_data/results/image_text_test_dataset.json', 'w') as f:
    json.dump(image_text_test_dataset, f)
