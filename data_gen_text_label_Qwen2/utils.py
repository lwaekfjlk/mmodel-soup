import numpy as np
import litellm
from typing import Dict, List
import json
from sklearn.metrics import f1_score
from scipy.special import softmax

def prompt_llm(messages) -> Dict[str, int]:
    for _ in range(5):
        response = litellm.completion(
            model='openai/Qwen2-72B-Instruct',
            messages=messages,
            base_url='http://cccxc713.pok.ibm.com:8000/v1',
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
    return {'logits': None}

def save_results(results: Dict, save_file: str):
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=4)

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