import json
import sys
from collections import Counter

sys.path.append('../')
from utils import read_json_file, save_dataset, construct_subset

def read_preds():
    text_only_pred = read_json_file('../urfunny_data/data_gen_output/urfunny_text_only_pred_qwen2.json')
    vision_only_pred = read_json_file('../urfunny_data/data_gen_output/urfunny_image_only_pred_cogvlm2.json')
    return text_only_pred, vision_only_pred 

def read_groundtruth_labels(split):
    with open(f'../urfunny_data/data_raw/urfunny_dataset_{split}.json', 'r') as file:
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

def record_label_distribution(ids, label_dict):
    return Counter([label_dict[id] for id in ids])

def main():
    for split in ['train', 'test', 'val']:
        gth_label = read_groundtruth_labels(split)
        text_only_pred, vision_only_pred = read_preds()
        R_ids, AS_ids, U_ids = select_subset_ids(text_only_pred, vision_only_pred, gth_label)
        
        train_dataset = read_json_file(f'../urfunny_data/data_raw/urfunny_dataset_{split}.json')
        
        R_dataset = construct_subset(R_ids, train_dataset)
        save_dataset(f'../urfunny_data/data_split_output/urfunny_R_dataset_{split}_cogvlm2_qwen2.json', R_dataset)
        
        AS_dataset = construct_subset(AS_ids, train_dataset)
        save_dataset(f'../urfunny_data/data_split_output/urfunny_AS_dataset_{split}_cogvlm2_qwen2.json', AS_dataset)
        
        U_dataset = construct_subset(U_ids, train_dataset)
        save_dataset(f'../urfunny_data/data_split_output/urfunny_U_dataset_{split}_cogvlm2_qwen2.json', U_dataset)
        
        R_label_distribution = record_label_distribution(R_ids, gth_label)
        AS_label_distribution = record_label_distribution(AS_ids, gth_label)
        U_label_distribution = record_label_distribution(U_ids, gth_label)
        
        print(split)
        print(f"R_ids: {len(R_ids)}")
        print(f"AS_ids: {len(AS_ids)}")
        print(f"U_ids: {len(U_ids)}")
        print("Ground-truth label distribution for R_ids:", R_label_distribution)
        print("Ground-truth label distribution for AS_ids:", AS_label_distribution)
        print("Ground-truth label distribution for U_ids:", U_label_distribution)

        # show prediction distribution
        text_only_pred_distribution = Counter([value['pred'] for value in text_only_pred.values()])
        vision_only_pred_distribution = Counter([value['pred'] for value in vision_only_pred.values()])
        print("Text-only prediction distribution:", text_only_pred_distribution)
        print("Vision-only prediction distribution:", vision_only_pred_distribution)

if __name__ == "__main__":
    main()
