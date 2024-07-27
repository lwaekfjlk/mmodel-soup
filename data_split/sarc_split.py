import json
import sys
sys.path.append('../')
from utils import read_json_file, save_dataset, construct_subset


def read_preds():
    text_only_pred = read_json_file('../sarc_data/data_gen_output/sarc_text_only_pred_qwen2.json')
    vision_only_pred = read_json_file('../sarc_data/data_gen_output/sarc_image_only_pred_cogvlm2.json')
    return text_only_pred, vision_only_pred 

def read_groundtruth_labels(split):
    with open(f'../sarc_data/data_raw/sarc_dataset_{split}.json', 'r') as file:
        dataset = json.load(file)
        return {key: value['label'] for key, value in dataset.items()}

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
    for split in ['train', 'test', 'val']:
        gth_label = read_groundtruth_labels(split)
        text_only_pred, vision_only_pred = read_preds()
        R_ids, AS_ids, U_ids = select_subset_ids(text_only_pred, vision_only_pred, gth_label)
        
        train_dataset = read_json_file(f'../sarc_data/data_raw/sarc_dataset_{split}.json')
        
        R_dataset = construct_subset(R_ids, train_dataset)
        save_dataset(f'../sarc_data/data_split_output/sarc_R_dataset_{split}_cogvlm2_qwen2.json', R_dataset)
        
        AS_dataset = construct_subset(AS_ids, train_dataset)
        save_dataset(f'../sarc_data/data_split_output/sarc_AS_dataset_{split}_cogvlm2_qwen2.json', AS_dataset)
        
        U_dataset = construct_subset(U_ids, train_dataset)
        save_dataset(f'../sarc_data/data_split_output/sarc_U_dataset_{split}_cogvlm2_qwen2.json', U_dataset)
        
        print(split)
        print(f"R_ids: {len(R_ids)}")
        print(f"AS_ids: {len(AS_ids)}")
        print(f"U_ids: {len(U_ids)}")

if __name__ == "__main__":
    main()