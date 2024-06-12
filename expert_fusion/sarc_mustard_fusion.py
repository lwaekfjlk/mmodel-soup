import numpy as np
from collections import defaultdict
import os
import json
import jsonlines
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
                image_id, text = line['image_id'], ''
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
                image_id, text = line['image_id'], ''
                data_id = f"{image_id}"
                dataset[name].append(line)
                results[data_id]['logits'][name] = line['logits']
                if results[data_id]['target'] is None:
                    results[data_id]['target'] = line['target']
                assert results[data_id]['target'] == line['target'], "Targets do not match across subsets for the same data."
    return dataset, results

def interaction_type_acc(results, interaction_type='AS'):
    gths = []
    preds = []
    for data_id, data in results.items():
        total_logits = data['logits'][interaction_type]
        predicted_label = total_logits.index(max(total_logits))
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy

def simple_average_fusion(results):
    gths = []
    preds = []
    with open('./blip2_fuser/test_rus_logits.json', 'r') as f:
        rus_logits = json.load(f)

    for data_id, data in results.items():
        total_logits = [sum(logits) / len(logits) for logits in zip(*data['logits'].values())]
        #total_logits = [0, 0]
        #for i in range(2):
        #    total_logits[i] = data['logits']['R'][i] * rus_logits[data_id][0] + data['logits']['U'][i] * rus_logits[data_id][1] + data['logits']['AS'][i] * rus_logits[data_id][2]
        predicted_label = total_logits.index(max(total_logits))
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy

def weighted_average_fusion(results, weights):
    gths = []
    preds = []
    for data_id, data in results.items():
        weighted_logits = [sum(w * logits[i] for w, logits in zip(weights, data['logits'].values()))
                           for i in range(len(next(iter(data['logits'].values()))))]
        predicted_label = weighted_logits.index(max(weighted_logits))
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy

def max_fusion(results):
    gths = []
    preds = []
    for data_id, data in results.items():
        max_logits = [max(logits) for logits in zip(*data['logits'].values())]
        predicted_label = max_logits.index(max(max_logits))
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy

def standardize(logits):
    mean = np.mean(logits, axis=0)
    std = np.std(logits, axis=0)
    return (logits - mean) / std

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def softmax_fusion(results):
    gths = []
    preds = []
    softmaxed_probs_data = {}
    # normalize the distribution based on all the logits

    with open('./blip2_fuser/test_rus_logits.json', 'r') as f:
        rus_logits = json.load(f)
    
    #for data_id, data in rus_logits.items():
    #    rus_logits[data_id] = np.exp(data) / np.sum(np.exp(data))

    for data_id, data in results.items():
        softmaxed_probs = [np.exp(logits) / np.sum(np.exp(logits)) for logits in data['logits'].values()]
        average_probs = softmaxed_probs[0] * rus_logits[data_id][0] + softmaxed_probs[1] * rus_logits[data_id][1] + softmaxed_probs[2] * rus_logits[data_id][2]
        #average_probs = np.mean(softmaxed_probs, axis=0)
        predicted_label = np.argmax(average_probs)
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy

def cascaded_fusion(results, threshold):
    gths = []
    preds = []
    softmaxed_probs = {}
    for data_id, data in results.items():
        for interaction_type, logits in data['logits'].items():
            softmaxed_probs[interaction_type] = np.exp(logits) / np.sum(np.exp(logits))
        if np.max(softmaxed_probs['R']) > threshold and np.max(softmaxed_probs['U']) > threshold:
            predicted_label = np.argmax(softmaxed_probs['R']) if np.max(softmaxed_probs['R']) > np.max(softmaxed_probs['U']) else np.argmax(softmaxed_probs['U'])
        else:
            predicted_label = np.argmax(softmaxed_probs['AS'])
        gths.append(data['target'])
        preds.append(predicted_label)
    f1, precision, recall, accuracy = f1_score(gths, preds), precision_score(gths, preds), recall_score(gths, preds), accuracy_score(gths, preds)
    return f1, precision, recall, accuracy


# Example usage within your main workflow
if __name__ == "__main__":
    file_dir = '../sarc_mustard_mixed/expert_blip2'
    _, transformed_results = load_and_transform_data(file_dir)
    weights = {'AS': 0.4, 'R': 0.2, 'U': 0.2}
    weighted_weights = [weights[name] for name in ['AS', 'R', 'U']]

    print("Simple Average Fusion Accuracy:", simple_average_fusion(transformed_results))
    print("Weighted Average Fusion Accuracy:", weighted_average_fusion(transformed_results, weighted_weights))
    print("Max Fusion Accuracy:", max_fusion(transformed_results))
    print("Softmax Fusion Accuracy:", softmax_fusion(transformed_results))
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"Cascaded Fusion Accuracy (Threshold={threshold}):", cascaded_fusion(transformed_results, threshold))
    print("AS Interaction Type Accuracy:", interaction_type_acc(transformed_results, 'AS'))
    print("R Interaction Type Accuracy:", interaction_type_acc(transformed_results, 'R'))
    print("U Interaction Type Accuracy:", interaction_type_acc(transformed_results, 'U'))
    baseline_dataset, baseline_results = load_and_transform_baseline(file_dir)
    print("Baseline Interaction Type Accuracy:", interaction_type_acc(baseline_results, 'baseline'))