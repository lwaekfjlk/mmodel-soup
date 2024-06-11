import json
import sys
sys.path.append('../')
from utils import read_json_file, save_dataset, construct_subset


def read_preds(task):
    preds_idiom = read_json_file(f'../../irfl_data/data_gen_output/irfl_idiom_pred.json')
    preds_metaphor = read_json_file(f'../../irfl_data/data_gen_output/irfl_metaphor_pred.json')
    preds_simile = read_json_file(f'../../irfl_data/data_gen_output/irfl_simile_pred.json')
    preds = {}
    preds.update(preds_idiom)
    preds.update(preds_metaphor)
    preds.update(preds_simile)
    return preds

def read_groundtruth_labels(task):
    with open(f'../../irfl_data/data_raw/irfl_dataset_train.json', 'r') as file:
        dataset = json.load(file)
        return {key: 1 if 'Figurative' in value['category'] else 0 for key, value in dataset.items()}

def select_subset_ids(preds, labels):
    R_ids, AS_ids, U_ids = [], [], []
    delta = 0.3
    for id, pred in preds.items():
        for text, value in pred.items():
            logits = value['logits']
            pred = value['pred']
            import pdb; pdb.set_trace()
            label = labels[id]
            if pred == label and abs(logits['Yes'] - logits['No']) > delta:
                R_ids.append(id)
            if pred != label and abs(logits['Yes'] - logits['No']) > delta:
                AS_ids.append(id)
            if abs(logits['Yes'] - logits['No']) <= delta:
                U_ids.append(id)
    return R_ids, AS_ids, U_ids

def main():
    tasks = ['simile', 'metaphor', 'idiom']
    for task in tasks:
        gth_label = read_groundtruth_labels(task)
        preds = read_preds(task)
        R_ids, AS_ids, U_ids = select_subset_ids(preds, gth_label)
        
        train_dataset = read_json_file(f'../../irfl_data/data_raw/irfl_dataset_train.json')
        
        R_dataset = construct_subset(R_ids, train_dataset)
        save_dataset(f'../../irfl_data/data_split_output/irfl_R_dataset_train.json', R_dataset)
        
        AS_dataset = construct_subset(AS_ids, train_dataset)
        save_dataset(f'../../irfl_data/data_split_output/irfl_AS_dataset_train.json', AS_dataset)
        
        U_dataset = construct_subset(U_ids, train_dataset)
        save_dataset(f'../../irfl_data/data_split_output/irfl_U_dataset_train.json', U_dataset)
        
        print(task)
        print(f"R_ids: {len(R_ids)}")
        print(f"AS_ids: {len(AS_ids)}")
        print(f"U_ids: {len(U_ids)}")


if __name__ == "__main__":
    main()
