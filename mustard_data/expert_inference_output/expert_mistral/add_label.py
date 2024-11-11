import os
import jsonlines   

subset_names = ['AS', 'R', 'U']
file_dir = '/mustard_data/expert_inference_output/expert_mistral'

for name in subset_names:
    out = {}
    file_path = os.path.join(file_dir, f'{name}_yesno_logits.jsonl')
    checkpath =  os.path.join(file_dir, f'{name}_yesno_logits.jsonl')
    with jsonlines.open(file_path, 'r') as f:
        for line in f:
            out['image_id'] = line['image_id']
            out['logits'] = line['logits']


