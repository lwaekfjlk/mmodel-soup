import json
import sys
sys.path.append('../')
from utils import read_json_file, save_dataset, construct_subset


def read_preds():
    preds = read_json_file(f'../../nycartoon_data/data_gen_output/nycartoon_pred.json')
    return preds

def read_groundtruth_labels():
    with open(f'../../nycartoon_data/data_raw/nycartoon_dataset_train.json', 'r') as file:
        dataset = json.load(file)
        return {key: value['label'] for key, value in dataset.items()}

def select_subset_ids(preds, labels):
    R_ids, AS_ids, U_ids = [], [], []
    delta = 2
    for id, pred in preds.items():
        for text, value in pred.items():
            dataset_id = id + '_' + text
            logits = value['logits']
            pred = value['pred']
            label = labels[dataset_id]
            if pred == label and abs(logits['Yes'] - logits['No']) > delta:
                R_ids.append(dataset_id)
            if pred != label and abs(logits['Yes'] - logits['No']) > delta:
                AS_ids.append(dataset_id)
            if abs(logits['Yes'] - logits['No']) <= delta:
                U_ids.append(dataset_id)
    return R_ids, AS_ids, U_ids

def main():
    gth_label = read_groundtruth_labels()
    preds = read_preds()
    R_ids, AS_ids, U_ids = select_subset_ids(preds, gth_label)
    
    train_dataset = read_json_file(f'../../nycartoon_data/data_raw/nycartoon_dataset_train.json')
    
    R_dataset = construct_subset(R_ids, train_dataset)
    save_dataset(f'../../nycartoon_data/data_split_output/nycartoon_R_dataset_train.json', R_dataset)
    
    AS_dataset = construct_subset(AS_ids, train_dataset)
    save_dataset(f'../../nycartoon_data/data_split_output/nycartoon_AS_dataset_train.json', AS_dataset)
    
    U_dataset = construct_subset(U_ids, train_dataset)
    save_dataset(f'../../nycartoon_data/data_split_output/nycartoon_U_dataset_train.json', U_dataset)
    
    print(f"R_ids: {len(R_ids)}")
    print(f"AS_ids: {len(AS_ids)}")
    print(f"U_ids: {len(U_ids)}")


if __name__ == "__main__":
    main()
