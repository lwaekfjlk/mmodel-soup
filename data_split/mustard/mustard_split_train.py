import json
import sys
sys.path.append('../')
from utils import read_json_file, save_dataset, construct_subset


def read_preds():
    text_only_pred = read_json_file('../../mustard_data/data_gen_output/mustard_text_only_pred.json')
    vision_only_pred = read_json_file('../../mustard_data/data_gen_output/mustard_vision_only_pred.json')
    return text_only_pred, vision_only_pred 

def read_groundtruth_labels():
    with open('../../mustard_data/data_raw/mustard_dataset_train.json', 'r') as file:
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


def main():
    gth_label = read_groundtruth_labels()
    text_only_pred, vision_only_pred = read_preds()
    R_ids, AS_ids, U_ids = select_subset_ids(text_only_pred, vision_only_pred, gth_label)
    
    train_dataset = read_json_file('../../mustard_data/data_raw/mustard_dataset_train.json')
    
    R_dataset = construct_subset(R_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_R_dataset_train.json', R_dataset)
    
    AS_dataset = construct_subset(AS_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_AS_dataset_train.json', AS_dataset)
    
    U_dataset = construct_subset(U_ids, train_dataset)
    save_dataset('../../mustard_data/data_split_output/mustard_U_dataset_train.json', U_dataset)
    
    print(f"R_ids: {len(R_ids)}")
    print(f"AS_ids: {len(AS_ids)}")
    print(f"U_ids: {len(U_ids)}")

if __name__ == "__main__":
    main()