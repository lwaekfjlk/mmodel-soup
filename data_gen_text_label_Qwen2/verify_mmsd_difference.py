import json


with open('../mmsd_data/data_raw/val_data.json') as f:
    sarc_two_data = json.load(f)


with open('../sarc_data/data_raw/sarc_dataset_val.json') as f:
    sarc_data = json.load(f)

for k, v in sarc_two_data.items():
    sarc_data_example = sarc_data[k]
    sarc_two_data_example = v
    print(sarc_data_example['text'])
    assert sarc_data_example['text'] == sarc_two_data_example['text']