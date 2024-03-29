import json

with open('intermediate_data/ALBEF/sarc_U_vision_text_label.jsonl') as f:
    data = [json.loads(line) for line in f]
    
print(len(data))

# {"image_id": "905134014893195264", "text": "asked my hubby why <user> didn t have a terminal 2 he responded so the number <num> can be posted everywhere superstition \u2013 at toronto pearson international airport yyz", "pred": 1, "target": 1, "logits": [-1.298817753791809, 1.4017431735992432]}

# Calculate scores accuracy, precision, recall, f1

tp = 0
fp = 0
tn = 0
fn = 0

for d in data:
    if d['pred'] == 1 and d['target'] == 1:
        tp += 1
    elif d['pred'] == 1 and d['target'] == 0:
        fp += 1
    elif d['pred'] == 0 and d['target'] == 0:
        tn += 1
    elif d['pred'] == 0 and d['target'] == 1:
        fn += 1
        
accuracy = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)