import numpy as np
from collections import defaultdict
import os
import jsonlines
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_and_transform_baseline(file_dir):
    subset_names = ['baseline']
    dataset = defaultdict(list)
    results = defaultdict(lambda: {'logits': defaultdict(list), 'target': None})

    # Load data from files
    for name in subset_names:
        file_path = os.path.join(file_dir, f'mustard_{name}_logits.jsonl')
        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                image_id = line['image_id']
                data_id = f"{image_id}"
                dataset[name].append(line)
                results[data_id]['logits'][name] = line['logits']
                if results[data_id]['target'] is None:
                    results[data_id]['target'] = line['target']
                assert results[data_id]['target'] == line['target'], "Targets do not match across subsets for the same data."
    return dataset, results

def load_and_transform_data(file_dir):
    subset_names = ['AS', 'R', 'U']
    dataset = defaultdict(list)
    results = defaultdict(lambda: {'logits': defaultdict(list), 'target': None})

    # Load data from files
    for name in subset_names:
        file_path = os.path.join(file_dir, f'mustard_{name}_logits.jsonl')
        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                image_id = line['image_id']
                data_id = f"{image_id}"
                dataset[name].append(line)
                results[data_id]['logits'][name] = line['logits']
                if results[data_id]['target'] is None:
                    results[data_id]['target'] = line['target']
                assert results[data_id]['target'] == line['target'], "Targets do not match across subsets for the same data."
    return dataset, results

def interaction_type_acc(results, interaction_type='AS', model_type='AS'):
    gths = []
    preds = []

    with open('../mustard_data/data_split_output/mustard_AS_dataset_test.json', 'r') as f:
        AS_dataset_raw = json.load(f)
        AS_data_ids = list(AS_dataset_raw.keys())
    
    with open('../mustard_data/data_split_output/mustard_R_dataset_test.json', 'r') as f:
        R_dataset_raw = json.load(f)
        R_data_ids = list(R_dataset_raw.keys())

    with open('../mustard_data/data_split_output/mustard_U_dataset_test.json', 'r') as f:
        U_dataset_raw = json.load(f)
        U_data_ids = list(U_dataset_raw.keys())

    data_ids = {
        'AS': AS_data_ids,
        'R': R_data_ids,
        'U': U_data_ids,
        'baseline': AS_data_ids + R_data_ids + U_data_ids
    }

    for data_id, data in results.items():
        if data_id not in data_ids[interaction_type]:
            continue

        total_logits = data['logits'][model_type]
        predicted_label = total_logits.index(max(total_logits))
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy



# Example usage within your main workflow
if __name__ == "__main__":
    file_dir = '../sarc_mustard_mixed/expert_blip2'
    #file_dir = '../mustard_data/expert_inference_output/expert_albef'
    _, transformed_results = load_and_transform_data(file_dir)
    baseline_dataset, baseline_results = load_and_transform_baseline(file_dir)
    print("AS Interaction Type Accuracy:", interaction_type_acc(baseline_results, 'AS', 'baseline'))
    print("R Interaction Type Accuracy:", interaction_type_acc(baseline_results, 'R', 'baseline'))
    print("U Interaction Type Accuracy:", interaction_type_acc(baseline_results, 'U', 'baseline'))
    print("Baseline Interaction Type Accuracy:", interaction_type_acc(baseline_results, 'baseline', 'baseline'))