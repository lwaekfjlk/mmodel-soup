import json
import jsonlines
import torch.nn.functional as F
import torch

with jsonlines.open('../mustard_data/expert_inference_output/expert_blip2/mustard_baseline_logits.jsonl', 'r') as f:
    raw_logits = list(f)
    baseline_logits = {}
    for logit in raw_logits:
        baseline_logits[logit['image_id']] = F.softmax(torch.tensor(logit['logits']), dim=0).tolist()

with open('../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    R_dataset = json.load(f)
    test_R_ids = list(R_dataset.keys())

with open('../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    AS_dataset = json.load(f)
    test_AS_ids = list(AS_dataset.keys())

with open('../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2.json', 'r') as f:
    U_dataset = json.load(f)
    test_U_ids = list(U_dataset.keys())

R_logits = [max(data) for id, data in baseline_logits.items() if id in test_R_ids]
AS_logits = [max(data) for id, data in baseline_logits.items() if id in test_AS_ids]
U_logits = [max(data) for id, data in baseline_logits.items() if id in test_U_ids]

mean_R_logits = sum(R_logits) / len(R_logits)
mean_AS_logits = sum(AS_logits) / len(AS_logits)
mean_U_logits = sum(U_logits) / len(U_logits)

print(mean_R_logits)
print(mean_AS_logits)
print(mean_U_logits)

import pdb; pdb.set_trace()
