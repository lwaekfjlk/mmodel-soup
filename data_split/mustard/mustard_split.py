import json
from collections import defaultdict

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def parse_labels(reader):
    label_dict = defaultdict(lambda: {'yes': -float('inf'), 'no': -float('inf'), 'pred': None})
    for line in reader:
        identifier, json_str = line.split(' ', 1)
        data = json.loads(json_str)
        label_dict[identifier] = parse_prediction(data)
    return label_dict

def parse_prediction(data):
    yes_score = data.get('Yes', -float('inf'))
    no_score = data.get('No', -float('inf'))
    pred = 1 if yes_score > no_score else 0
    return {'yes': yes_score, 'no': no_score, 'pred': pred}

def read_vision_and_text_labels():
    text_label_reader = read_json_file('../../mustard_data/data_gen_output/mustard_text_only_pred.json')
    vision_label_reader = read_json_file('../../mustard_data/data_gen_output/mustard_vision_only_pred.json')

    import pdb; pdb.set_trace() 
    text_label_dict = parse_labels(text_label_reader)
    vision_label_dict = parse_labels(vision_label_reader)
    
    return text_label_dict, vision_label_dict

def read_groundtruth_labels():
    with open('../../mustard_data/data_raw/mustard_raw_data_speaker_independent_train.json', 'r') as file:
        dataset = json.load(file)
        return {key: 1 if value['sarcasm'] else 0 for key, value in dataset.items()}

def select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict):
    R_ids, AS_ids, U_ids = [], [], []
    for id, gth_label in gth_label_dict.items():
        text_label = text_label_dict.get(id, {}).get('pred')
        vision_label = vision_label_dict.get(id, {}).get('pred')
        if text_label is None or vision_label is None:
            continue
        if text_label == vision_label:
            (R_ids if text_label == gth_label else AS_ids).append(id)
        else:
            U_ids.append(id)
    return R_ids, AS_ids, U_ids

def construct_subset(ids, dataset):
    return {id: dataset[id] for id in ids if id in dataset}

def save_dataset(file_path, dataset):
    with open(file_path, 'w') as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    gth_label_dict = read_groundtruth_labels()
    text_label_dict, vision_label_dict = read_vision_and_text_labels()
    R_ids, AS_ids, U_ids = select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict)
    
    with open('../../mustard_data/data_raw/mustard_raw_data_speaker_independent_train.json', 'r') as file:
        train_dataset = json.load(file)
    
    with open('../../mustard_data/data_raw/mustard_raw_data_speaker_independent_test.json', 'r') as file:
        test_dataset = json.load(file)
    
    train_dataset = construct_subset(train_dataset.keys(), train_dataset)
    save_dataset('../../mustard_data/data_raw/mustard_dataset_train.json', train_dataset)

    test_dataset = construct_subset(test_dataset.keys(), test_dataset)
    save_dataset('../../mustard_data/data_raw/mustard_dataset_test.json', test_dataset)
    
    R_dataset = construct_subset(R_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_R_dataset_train.json', R_dataset)
    
    AS_dataset = construct_subset(AS_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_AS_dataset_train.json', AS_dataset)
    
    U_dataset = construct_subset(U_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_U_dataset_train.json', U_dataset)

    print(f"R_ids: {len(R_ids)}")
    print(f"AS_ids: {len(AS_ids)}")
    print(f"U_ids: {len(U_ids)}")
