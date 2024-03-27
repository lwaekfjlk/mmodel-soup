import ast
import json
from collections import OrderedDict
import pdb

def read_text_data(file_path):
    with open(file_path, "r") as f:
        input_str = f.read()
        lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
        # Convert each line from a string representation of a list to an actual list
        lists = [ast.literal_eval(line) for line in lines]
    return lists

train_data = read_text_data("text_data/train_filtered.txt")
# valid_data = read_text_data("text_data/valid.txt")
# test_data = read_text_data("text_data/test.txt")

labels = []
for data in [train_data]:
    for i in range(len(data)):
        image_name, image_class = data[i][0], data[i][2]
        labels.append((image_name, image_class))

with open("sarc_text_cls.txt", "r") as f:
    sarc_result = f.readlines()

logits_diffs = {}
yes_logprob, no_logprob = -float("inf"), -float("inf")
for i in range(len(sarc_result)):
    image_name, answer = sarc_result[i].split(" ", 1)
    logits = json.loads(answer, object_hook=OrderedDict)
    for token, logprob in logits.items():
        if token == 'Yes':
            yes_logprob = logprob
        if token == 'No':
            no_logprob = logprob
        # if 'yes' in token.lower():
        #     yes_logprob = max(yes_logprob, logprob)
        # elif 'no' in token.lower():
        #     no_logprob = max(no_logprob, logprob)
    logits_diff = yes_logprob - no_logprob
    logits_diffs[image_name] = logits_diff

# calculate accuracy, precision, recall, and F1 score
def calculate_metrics(labels, pred_map):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(labels)):
        image_name, image_class = labels[i]
        pred_class = pred_map[image_name] #pred_map["{}.jpg".format(image_name)]
        if image_class == 1 and pred_class == 1:
            tp += 1
        elif image_class == 0 and pred_class == 0:
            tn += 1
        elif image_class == 1 and pred_class == 0:
            fn += 1
        elif image_class == 0 and pred_class == 1:
            fp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0 
    return accuracy, precision, recall, f1

# find the threshold by grid search for the best F1 score
thresholds = [i/100 for i in range(-100, 100)]
best_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
best_threshold = 0

for threshold in thresholds:
    pred_map = {}
    for image_name, logits_diff in logits_diffs.items():
        pred_class = 1 if logits_diff > threshold else 0
        pred_map[image_name] = pred_class

    accuracy, precision, recall, f1 = calculate_metrics(labels, pred_map)
    if accuracy > best_metrics['accuracy']:
        best_metrics['accuracy'] = accuracy
        best_metrics['precision'] = precision
        best_metrics['recall'] = recall
        best_metrics['f1'] = f1
        best_threshold = threshold
        
print("Best threshold: ", best_threshold)
print("Best metrics: ", best_metrics)

# produce the final prediction into json file
pred_map = {}
for image_name, logits_diff in logits_diffs.items():
    pred_class = 1 if logits_diff > best_threshold else 0
    pred_map[image_name] = pred_class
    
import json
with open("sarc_text_pred.json", "w") as f:
    json.dump(pred_map, f)