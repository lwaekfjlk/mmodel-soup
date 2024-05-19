import json

with open('vision_text_label_logits.jsonl', 'r') as f:
    vision_text_label_logits = [json.loads(line) for line in f]

tp, fp, tn, fn = 0, 0, 0, 0
for item in vision_text_label_logits:
    if item['label'] == "Figurative" and item['response']['Yes'] > item['response']['No']:
        tp += 1
    elif item['label'] == "Figurative" and item['response']['Yes'] < item['response']['No']:
        fn += 1
    elif item['label'] != "Figurative" and item['response']['Yes'] > item['response']['No']:
        fp += 1
    elif item['label'] != "Figurative" and item['response']['Yes'] < item['response']['No']:
        tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (tp + tn) / (tp + fp + tn + fn)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
print("Accuracy: ", accuracy)