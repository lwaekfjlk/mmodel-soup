import numpy as np
import os
import jsonlines
import json
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_and_transform_data(dataset_name, file_dir, subset_names):
    results = defaultdict(lambda: {'logits': defaultdict(list), 'target': None})
    for name in subset_names:
        file_path = os.path.join(file_dir, f'{dataset_name}_{name}_logits.jsonl')
        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                data_id = line['image_id']
                results[data_id]['logits'][name] = line['logits']
                if results[data_id]['target'] is None:
                    results[data_id]['target'] = line['target']
                assert results[data_id]['target'] == line['target'], "Targets do not match across subsets for the same data."
    return results

def calculate_metrics(gths, preds):
    return {
        "f1": f1_score(gths, preds),
        "precision": precision_score(gths, preds),
        "recall": recall_score(gths, preds),
        "accuracy": accuracy_score(gths, preds)
    }

def get_predictions(results, fusion_strategy, *args):
    gths, preds = [], []
    for data_id, data in results.items():
        predicted_label = fusion_strategy(data['logits'], *args)
        gths.append(data['target'])
        preds.append(predicted_label)
    return calculate_metrics(gths, preds)

def simple_average(logits):
    avg_logits = np.mean([logits[name] for name in logits], axis=0)
    return np.argmax(avg_logits)

def weighted_average(logits, weights):
    weighted_logits = sum(weights[name] * np.array(logits[name]) for name in logits)
    return np.argmax(weighted_logits)

def max_fusion(logits):
    max_logits = np.max([logits[name] for name in logits], axis=0)
    return np.argmax(max_logits)

def softmax_fusion(logits):
    softmaxed_probs = np.mean([np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits], axis=0)
    return np.argmax(softmaxed_probs)

def cascaded_fusion(logits, threshold):
    softmaxed_probs = {name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()}
    if max(softmaxed_probs['R']) > threshold and max(softmaxed_probs['U']) > threshold:
        return np.argmax(softmaxed_probs['R']) if max(softmaxed_probs['R']) > max(softmaxed_probs['U']) else np.argmax(softmaxed_probs['U'])
    return np.argmax(softmaxed_probs['AS'])

def get_oracle_prediction(dataset_name, logits):
    fusion_type_dict = {}
    for interaction_type in ['AS', 'R', 'U']:
        with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json', 'r') as f:
            dataset = json.load(f)
        for image_id, data in dataset.items():
            fusion_type_dict[image_id] = interaction_type
    
    gths, preds = [], []
    for data_id, data in logits.items():
        interaction_type = fusion_type_dict[data_id]
        if interaction_type == 'AS':
            pred = np.argmax(data['logits']['AS'])
        elif interaction_type == 'R':
            pred = np.argmax(data['logits']['R'])
        elif interaction_type == 'U':
            pred = np.argmax(data['logits']['U'])
        gths.append(data['target'])
        preds.append(pred)
    return calculate_metrics(gths, preds) 


if __name__ == "__main__":
    dataset_name = 'mmsd'
    model_name = 'blip2'
    file_dir = f'../{dataset_name}_data/expert_inference_output/expert_{model_name}'
    
    subset_names = ['AS', 'R', 'U']
    results = load_and_transform_data(dataset_name, file_dir, subset_names)


    weights = {'AS': 0.0, 'R': 0.2, 'U': 0.2}
    weight_list = [weights[name] for name in subset_names]

    print("Oracle Prediction:", get_oracle_prediction(dataset_name, results))
    print("Simple Average Fusion:", get_predictions(results, simple_average))
    print("Weighted Average Fusion:", get_predictions(results, weighted_average, weights))
    print("Max Fusion:", get_predictions(results, max_fusion))
    print("Softmax Fusion:", get_predictions(results, softmax_fusion))
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"Cascaded Fusion (Threshold={threshold}):", get_predictions(results, cascaded_fusion, threshold))

    for interaction_type in subset_names:
        print(f"{interaction_type} Interaction Type Accuracy:", get_predictions(results, lambda x: np.argmax(x[interaction_type])))

    baseline_results = load_and_transform_data(dataset_name, file_dir, ['baseline'])
    print("Baseline Interaction Type Accuracy:", get_predictions(baseline_results, lambda x: np.argmax(x['baseline'])))
