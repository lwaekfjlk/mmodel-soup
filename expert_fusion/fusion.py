import numpy as np
import os
import jsonlines
import json
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

with open('../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    dataset = json.load(f)
    AS_test_data_ids = list(dataset.keys())

with open('../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    dataset = json.load(f)
    R_test_data_ids = list(dataset.keys())

with open('../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    dataset = json.load(f)
    U_test_data_ids = list(dataset.keys())


def load_weights(weights_file):
    with jsonlines.open(weights_file, 'r') as f:
        return {line['image_id']: line['logits'] for line in f}

def load_and_transform_data(dataset_name, file_dir, subset_names, weights=None):
    results = defaultdict(lambda: {'logits': defaultdict(list), 'target': None, 'weights': {}})
    for name in subset_names:
        file_path = os.path.join(file_dir, f'{dataset_name}_{name}_logits.jsonl')
        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                data_id = line['image_id']
                results[data_id]['logits'][name] = line['logits']
                results[data_id]['target'] = results[data_id]['target'] or line['target']
                assert results[data_id]['target'] == line['target'], "Targets do not match across subsets for the same data."
                if weights and data_id in weights:
                    results[data_id]['weights'][name] = weights[data_id][name]
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
    for id, data in results.items():
        predicted_label = fusion_strategy(data['logits'], data['weights'], *args)
        gths.append(data['target'])
        preds.append(predicted_label)
    return calculate_metrics(gths, preds)

def weighted_softmax_rus_fusion(logits, weights, *args):
    softmax_weights = {
        'R': np.exp(weights['R']) / (np.exp(weights['R']) + np.exp(weights['U']) + np.exp(weights['AS'])),
        'U': np.exp(weights['U']) / (np.exp(weights['R']) + np.exp(weights['U']) + np.exp(weights['AS'])),
        'AS': np.exp(weights['AS']) / (np.exp(weights['R']) + np.exp(weights['U']) + np.exp(weights['AS'])),
    }


    predicted_rus = np.argmax([softmax_weights['R'], softmax_weights['U'], softmax_weights['AS']])
    predicted_rus = 'pred R' if predicted_rus == 0 else 'pred U' if predicted_rus == 1 else 'pred AS'

    softmax_logits = {name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()}
    weighted_logits = sum(softmax_weights[name] * np.array(logit) for name, logit in softmax_logits.items())
    return np.argmax(weighted_logits)

def simple_average(logits, *args):
    sum_logits = np.mean([logits[name][:2] for name in logits], axis=0)
    return np.argmax(sum_logits)

def weighted_average(logits, weights, *args):
    weighted_logits = sum(weights[name] * np.array(logits[name]) for name in logits)
    return np.argmax(weighted_logits)

def max_fusion(logits, *args):
    max_logits = np.max([logits[name] for name in logits], axis=0)
    return np.argmax(max_logits)

def softmax_fusion(logits, *args):
    softmaxed_probs = np.mean([np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits], axis=0)
    return np.argmax(softmaxed_probs)

def cascaded_fusion(logits, threshold, *args):
    softmaxed_probs = {name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()}
    if max(softmaxed_probs['R']) > threshold and max(softmaxed_probs['U']) > threshold:
        return np.argmax(softmaxed_probs['R'] if max(softmaxed_probs['R']) > max(softmaxed_probs['U']) else softmaxed_probs['U'])
    return np.argmax(softmaxed_probs['AS'])

def get_oracle_prediction(dataset_name, logits):
    fusion_type_dict = {}
    for interaction_type in ['AS', 'R', 'U']:
        with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json', 'r') as f:
            dataset = json.load(f)
        for image_id in dataset:
            fusion_type_dict[image_id] = interaction_type

    gths, preds = [], []
    for data_id, data in logits.items():
        interaction_type = fusion_type_dict[data_id]
        pred = np.argmax(data['logits'][interaction_type])
        gths.append(data['target'])
        preds.append(pred)
    return calculate_metrics(gths, preds)

def use_unimodal_label_for_prediction(dataset_name, logits):
    fusion_type_dict = {}
    gths = []
    preds = []

    for interaction_type in ['AS', 'R', 'U']:
        with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json', 'r') as f:
            dataset = json.load(f)
        for image_id in dataset:
            fusion_type_dict[image_id] = interaction_type

    with open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.json', 'r') as f:
        image_only_labels = json.load(f)

    with open(f'../{dataset_name}_data/data_gen_output/{dataset_name}_text_only_pred_qwen2.json', 'r') as f:
        text_only_labels = json.load(f)

    for id, data in logits.items():
        image_only_label = image_only_labels[id]['pred']
        text_only_label = text_only_labels[id]['pred']
        logits = data['logits']
        softmax_probs = {name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()}
        if image_only_label == text_only_label:
            R_logits = softmax_probs['R']
            AS_logits = softmax_probs['AS']
            pred = np.argmax([R_logits[0]+AS_logits[0], R_logits[1]+AS_logits[1]])
        else:
            pred = np.argmax(logits['U'])
        gths.append(data['target'])
        preds.append(pred)
    return calculate_metrics(gths, preds)


def main():
    dataset_name = 'mmsd'
    model_name = 'phi3v'

    with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_AS_dataset_test_cogvlm2_qwen2.json', 'r') as f:
        dataset = json.load(f)
    AS_test_data_ids = list(dataset.keys())

    with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_R_dataset_test_cogvlm2_qwen2.json', 'r') as f:
        dataset = json.load(f)
    R_test_data_ids = list(dataset.keys())

    with open(f'../{dataset_name}_data/data_split_output/{dataset_name}_U_dataset_test_cogvlm2_qwen2.json', 'r') as f:
        dataset = json.load(f)
    U_test_data_ids = list(dataset.keys())

    file_dir = f'../{dataset_name}_data/expert_inference_output/expert_{model_name}'
    weights_file = f'../{dataset_name}_data/expert_inference_output/expert_{model_name}/{dataset_name}_rus_logits.jsonl'
    subset_names = ['R', 'U', 'AS']

    weights = load_weights(weights_file) if os.path.exists(weights_file) else None
    results = load_and_transform_data(dataset_name, file_dir, subset_names, weights)

    baseline_results = load_and_transform_data(dataset_name, file_dir, ['baseline'], {})

    print("Baseline Interaction Type Accuracy:", get_predictions(baseline_results, lambda x, y: np.argmax(x['baseline'])))
    print("Simple Average Fusion:", get_predictions(results, simple_average))
    if weights:
        print("RUS Fusion:", get_predictions(results, weighted_softmax_rus_fusion))

    print("Oracle Prediction:", get_oracle_prediction(dataset_name, results))
    print("Unimodal labels augmented prediction", use_unimodal_label_for_prediction(dataset_name, results))
    print("Max Fusion:", get_predictions(results, max_fusion))
    print("Softmax Fusion:", get_predictions(results, softmax_fusion))

    subpart_results = {'AS': {}, 'R': {}, 'U': {}}
    subpart_baseline_results = {'AS': {}, 'R': {}, 'U': {}}
    for result in results:
        if result in AS_test_data_ids:
            subpart_results['AS'][result] = results[result]
            subpart_baseline_results['AS'][result] = baseline_results[result]
        elif result in R_test_data_ids:
            subpart_results['R'][result] = results[result]
            subpart_baseline_results['R'][result] = baseline_results[result]
        elif result in U_test_data_ids:
            subpart_results['U'][result] = results[result]
            subpart_baseline_results['U'][result] = baseline_results[result]

    for interaction_type in subset_names:
        print(f"{interaction_type} expert results on the {interaction_type} test set:", 
              get_predictions(subpart_results[interaction_type], lambda x, y: np.argmax(x[interaction_type])))
        print(f"Baseline results on the {interaction_type} test set:", 
              get_predictions(subpart_baseline_results[interaction_type], lambda x, y: np.argmax(x['baseline'])))
        print(f"{interaction_type} expert results on the whole test set:",
              get_predictions(results, lambda x, y: np.argmax(x[interaction_type])))
        print("="*10)

if __name__ == "__main__":
    main()
