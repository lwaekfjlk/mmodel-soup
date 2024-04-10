import json
from collections import defaultdict

def read_vision_and_text_labels():
    text_label_dict = defaultdict(dict)
    vision_label_dict = defaultdict(dict)
    with open('../../mustard_data/results/mustard_text_label_test.json', 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, json_str = line.split(' ', 1)
            data = json.loads(json_str)
            text_label_dict[identifier] = {'yes': -float('inf'), 'no': -float('inf')}
            if 'Yes' in data.keys() and 'No' in data.keys():
                text_label_dict[identifier]['yes'] = data['Yes']
                text_label_dict[identifier]['no'] = data['No']
                text_label_dict[identifier]['pred'] = 1 if data['Yes'] > data['No'] else 0
            if 'Yes' in data.keys() and 'No' not in data.keys():
                text_label_dict[identifier]['yes'] = data['Yes']
                text_label_dict[identifier]['no'] = -float('inf')
                text_label_dict[identifier]['pred'] = 1
            if 'Yes' not in data.keys() and 'No' in data.keys():
                text_label_dict[identifier]['yes'] = -float('inf')
                text_label_dict[identifier]['no'] = data['No']
                text_label_dict[identifier]['pred'] = 0

    with open('../../mustard_data/results/mustard_vision_label_test.json', 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, list_str = line.split(' ', 1)
            data = json.loads(json_str)
            vision_label_dict[identifier] = {'yes': -float('inf'), 'no': -float('inf')}
            if 'Yes' in data.keys() and 'No' in data.keys():
                vision_label_dict[identifier]['yes'] = data['Yes']
                vision_label_dict[identifier]['no'] = data['No']
                vision_label_dict[identifier]['pred'] = 1 if data['Yes'] > data['No'] else 0
            if 'Yes' in data.keys() and 'No' not in data.keys():
                vision_label_dict[identifier]['yes'] = data['Yes']
                vision_label_dict[identifier]['no'] = -float('inf')
                vision_label_dict[identifier]['pred'] = 1
            if 'Yes' not in data.keys() and 'No' in data.keys():
                vision_label_dict[identifier]['yes'] = -float('inf')
                vision_label_dict[identifier]['no'] = data['No']
                vision_label_dict[identifier]['pred'] = 0
    return text_label_dict, vision_label_dict


def read_groundtruth_labels():
    gth_label_dict = {}
    with open('../../mustard_data/intermediate_data/sarcasm_data_speaker_independent_train.json', 'r') as f:
        dataset = json.load(f)
        for key, value in dataset.items():
            gth_label_dict[key] = 1 if value['sarcasm'] == True else 0
    return gth_label_dict


def select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict):
    R_ids = []
    AS_ids = []
    U_ids = []
    for id, gth_label in gth_label_dict.items():
        try:
            text_label = text_label_dict[id]['pred']
            vision_label = vision_label_dict[id]['pred']
            if text_label == vision_label and text_label == gth_label:
                R_ids.append(id)
            if text_label == vision_label and text_label != gth_label:
                AS_ids.append(id)
            if text_label != vision_label:
                U_ids.append(id)
        except:
            if 'pred' not in text_label_dict[id].keys():
                print('no text_label')
            if 'pred' not in vision_label_dict[id].keys():
                print('no vision_label')
            continue
    return R_ids, AS_ids, U_ids


def construct_subset(ids):
    subset = {}
    with open('../../mustard_data/intermediate_data/sarcasm_data_speaker_independent_train.json', 'r') as f:
        dataset = json.load(f)
        for id in ids:
            subset[id] = dataset[id]
    return subset


if __name__ == "__main__":
    gth_label_dict = read_groundtruth_labels()
    text_label_dict, vision_label_dict = read_vision_and_text_labels()
    R_ids, AS_ids, U_ids = select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict)
    R_dataset = construct_subset(R_ids)
    with open('../../mustard_data/results/mustard_R_dataset_train.json', 'w') as f:
        json.dump(R_dataset, f)
    AS_dataset = construct_subset(AS_ids)
    with open('../../mustard_data/results/mustard_AS_dataset_train.json', 'w') as f:
        json.dump(AS_dataset, f)
    U_dataset = construct_subset(U_ids)
    with open('../../mustard_data/results/mustard_U_dataset_train.json', 'w') as f:
        json.dump(U_dataset, f)

    print(len(R_ids))
    print(len(AS_ids))
    print(len(U_ids))

