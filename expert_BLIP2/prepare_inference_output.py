import json
import os
import jsonlines

dataset_name = 'mmsd'

inference_results = [f'{dataset_name}_baseline_logits.jsonl', f'{dataset_name}_AS_logits.jsonl', f'{dataset_name}_R_logits.jsonl', f'{dataset_name}_U_logits.jsonl']
model_directories = [f'blip2_{dataset_name}_baseline_model', f'blip2_{dataset_name}_AS_model', f'blip2_{dataset_name}_R_model', f'blip2_{dataset_name}_U_model']


for inference_result, model_directory in zip(inference_results, model_directories):
    overall_dataset = []

    gth = {}
    with open(f'../{dataset_name}_data/data_raw/{dataset_name}_dataset_test.json', 'r') as f:
        dataset = json.load(f)
        for image_id, data in dataset.items():
            if dataset_name == 'mustard':
                gth[image_id] = 1 if data['sarcasm'] else 0
            else:
                gth[image_id] = data['label']

    with open(f'./{model_directory}/test_yesno_logits.json', 'r') as f:
        inference_output = json.load(f)
        for image_id, logits in inference_output.items():
            overall_dataset.append({'image_id': image_id, 'logits': logits, 'target': gth[image_id]})

    # create expert_inference_output expert_blip2 directory if it does not exist
    if not os.path.exists(f'../{dataset_name}_data/expert_inference_output/expert_blip2'):
        os.makedirs(f'../{dataset_name}_data/expert_inference_output/expert_blip2')

    with jsonlines.open(f'../{dataset_name}_data/expert_inference_output/expert_blip2/{inference_result}', 'w') as f:
        f.write_all(overall_dataset)