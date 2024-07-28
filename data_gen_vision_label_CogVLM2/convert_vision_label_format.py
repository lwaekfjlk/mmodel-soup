import jsonlines
from typing import Dict, List
import numpy as np
import json
from sklearn.metrics import f1_score
from scipy.special import softmax
from collections import Counter

def apply_thresholds(results: Dict, thresholds: List[float]) -> Dict[float, float]:
    f1_scores = {}
    for threshold in thresholds:
        y_true, y_pred = [], []
        for value in results.values():
            if value['logits'] and value['gth'] is not None:
                prob_yes = value['logits']['Yes']
                y_true.append(value['gth'])
                y_pred.append(1 if prob_yes >= threshold else 0)
        f1_scores[threshold] = f1_score(y_true, y_pred)
    return f1_scores

def add_pred_based_on_threshold(results: Dict, threshold: float) -> Dict:
    y_pred = {}
    for key, value in results.items():
        if value['logits'] and value['gth'] is not None:
            prob_yes = value['logits']['Yes']
            y_pred[key] = 1 if prob_yes >= threshold else 0
            results[key]['pred'] = y_pred[key]
        else:
            y_pred[key] = None
            results[key]['pred'] = None
    new_results = {}
    for key, result in results.items():
        if result['gth'] is None:
            continue
        new_results[key] = result
    return new_results

def save_results_to_json(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def select_top_percent_as_one(results, percentage):
    logits = [(image_id, data['logits']) for image_id, data in results.items()]
    sorted_logits = sorted(logits, key=lambda x: x[1]['Yes'], reverse=True)
    threshold_index = int(len(sorted_logits) * percentage)

    for i, (image_id, logit) in enumerate(sorted_logits):
        if i < threshold_index:
            results[image_id]['pred'] = 1
        else:
            results[image_id]['pred'] = 0
    return results

if __name__ == "__main__":

    dataset_name = 'mustard'

    with jsonlines.open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.jsonl') as reader:
        dataset = list(reader)

    gth_label_count = Counter([value['gth'] for value in dataset if value['gth'] is not None])
    yes_percentage = gth_label_count[1] / sum(gth_label_count.values())
    print(f'Yes percentage in gth: {yes_percentage}')

    results = {}
    for data in dataset:
        image_id = data['image_id']
        # normalize logits
        results[image_id] = {
            'logits': data['logits'],
            'gth': data['gth'],
        }

    results = select_top_percent_as_one(results, yes_percentage)

    preds = [value['pred'] for value in results.values() if value['gth'] is not None]
    gths = [value['gth'] for value in results.values() if value['gth'] is not None]

    f1 = f1_score(gths, preds)
    print(f'F1 score: {f1}')

    with open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.json', 'w') as f:
        json.dump(results, f, indent=4)
