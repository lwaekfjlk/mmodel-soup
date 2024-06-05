import json
with open('expert_albef/irfl_R_logits.jsonl') as f:
    data = [json.loads(line) for line in f]

id_to_label = {d['image_id']: d['target'] for d in data}
# print(len(id_to_label))
# print(len(data))

with open("expert_blip2/U_yesno_logits.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

preds, targets = [], []
for d in data:
    pred = 1 if d['logits'][0] > d['logits'][1] else 0
    target = id_to_label[d['image_id']]
    preds.append(pred)
    targets.append(target)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
accuracy = accuracy_score(targets, preds)
f1 = f1_score(targets, preds)
precision = precision_score(targets, preds)
recall = recall_score(targets, preds)
print(f'{accuracy:.4f}')
print(f'{f1:.4f}')
print(f'{precision:.4f}')
print(f'{recall:.4f}')