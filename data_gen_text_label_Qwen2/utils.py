import numpy as np
import litellm
from typing import Dict, List
import concurrent.futures
from tqdm import tqdm
import json
from sklearn.metrics import f1_score
from scipy.special import softmax

def prompt_llm(messages) -> Dict[str, int]:
    for _ in range(5):
        response = litellm.completion(
            model='openai/Qwen2-72B-Instruct',
            messages=messages,
            base_url='http://cccxc713.pok.ibm.com:8001/v1',
            api_key='fake',
            temperature=0,
            top_p=1,
            logprobs=True,
            top_logprobs=5,
        )
        top_logprobs = response.choices[0].logprobs['content'][0]['top_logprobs']
        res = {logprob['token']: logprob['logprob'] for logprob in top_logprobs}

        if 'Yes' in res and 'No' in res:
            return {'logits': {'Yes': np.exp(res['Yes']), 'No': np.exp(res['No'])}}
        elif 'Yes' in res and 'No' not in res:
            return {'logits': {'Yes': np.exp(res['Yes']), 'No': -10000}}
        elif 'Yes' not in res and 'No' in res:
            return {'logits': {'Yes': -10000, 'No': np.exp(res['No'])}}
    return {'logits': {'Yes': -10000, 'No': -10000}}

def save_results(results: Dict, save_file: str):
    filtered_results = {}
    for key, value in results.items():
        if value['logits'] and 'Yes' in value['logits'] and 'No' in value['logits']:
            filtered_results[key] = value
    with open(save_file, 'w') as f:
        json.dump(filtered_results, f, indent=4)

def apply_thresholds(results: Dict, thresholds: List[float]) -> Dict[float, float]:
    f1_scores = {}
    for threshold in thresholds:
        y_true, y_pred = [], []
        for value in results.values():
            if value['logits']:
                prob_yes = softmax([value['logits']['Yes'], value['logits']['No']])[0]
                y_true.append(value['gth'])
                y_pred.append(1 if prob_yes >= threshold else 0)
        f1_scores[threshold] = f1_score(y_true, y_pred)
    return f1_scores

def add_pred_based_on_threshold(results: Dict, threshold: float) -> Dict:
    y_pred = {}
    for key, value in results.items():
        if value['logits']:
            prob_yes = softmax([value['logits']['Yes'], value['logits']['No']])[0]
            y_pred[key] = 1 if prob_yes >= threshold else 0
            results[key]['pred'] = y_pred[key]
        else:
            y_pred[key] = None
            results[key]['pred'] = None
    return results

def multi_process_run(process_text, results, dataset, max_workers, save_file):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(process_text, value): key for key, value in dataset.items() if key not in results or results[key]['logits'] is None}
        print(future_to_key)
        for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc='Processing'):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {'logits': None, 'gth': dataset[key]['label']}
                print(f"Error processing {key}: {e}")
            save_results(results, save_file)


def select_top_percent_as_one(results, percentage):
    logits = [(image_id, data['logits']) for image_id, data in results.items()]
    sorted_logits = sorted(logits, key=lambda x: x[1]['Yes'], reverse=True)
    threshold_index = int(len(sorted_logits) * percentage)

    for i, (image_id, logit) in enumerate(sorted_logits):
        if i < threshold_index:
            results[image_id]['pred'] = 1
        else:
            results[image_id]['pred'] = 0
    return results