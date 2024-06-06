import json
with open('expert_albef/mustard_U_logits.jsonl') as f:
    data = [json.loads(line) for line in f]

data = data

preds, targets = [], []
for d in data:
    preds.append(d['pred'])
    targets.append(d['target'])

# calculate Accuracy F1 Precision Recall
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
accuracy = accuracy_score(targets, preds)
f1 = f1_score(targets, preds)
precision = precision_score(targets, preds)
recall = recall_score(targets, preds)

print(f'{accuracy:.4f}')
print(f'{f1:.4f}')
print(f'{precision:.4f}')
print(f'{recall:.4f}')