import jsonlines
from typing import Dict, List
import numpy as np
import json
from sklearn.metrics import f1_score
from scipy.special import softmax

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

if __name__ == "__main__":

    dataset_name = 'urfunny'

    with jsonlines.open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.jsonl') as reader:
        dataset = list(reader)

    results = {}
    for data in dataset:
        image_id = data['image_id']
        results[image_id] = {
            'logits': data['logits'],
            'gth': data['gth'],
        }

    thresholds = np.arange(0., 1.0, 0.02)
    f1_scores = apply_thresholds(results, thresholds)
    best_threshold = max(f1_scores, key=f1_scores.get)
    print(f"Best Threshold: {best_threshold}, Best F1 Score: {f1_scores[best_threshold]}")

    results = add_pred_based_on_threshold(results, best_threshold)

    with open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.json', 'w') as f:
        json.dump(results, f, indent=4)
