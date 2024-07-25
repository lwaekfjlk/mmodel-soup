import json
import argparse
from typing import List, Dict
from tqdm import tqdm
import litellm
import concurrent.futures
import os
import numpy as np
from sklearn.metrics import f1_score
from utils import prompt_llm, save_results, apply_thresholds, add_pred_based_on_threshold

SYS_PROMPT = (
    "Please analyze the text provided below for humor or not."
    "If you think the text includes exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'."
    "If you think the text is neutral or just common meaning, please answer 'No'."
    "Please make sure that your answer is based on the text itself, not on the context or your personal knowledge."
    "There are only two options: 'Yes' or 'No'."
    "If you are not sure, please provide your best guess and do not say that you are not sure."
    "You should only make No judgement when you are very sure that the text is not funny. As long as you think potentially it is funny, you should say Yes."
)

EXAMPLES = [
    {"role": "user", "content": "TEXT: why invite men they are the problem"},
    {"role": "assistant", "content": "Yes. It expresses that men can be problematic and the speaker is sarcastic to make people laugh."},
    {"role": "user", "content": "TEXT: we all feel the same things."},
    {"role": "assistant", "content": "No. It is a neutral statement."}
]


def process_text(data_item: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    text = 'TEXT: ' + data_item['punchline_sentence']
    messages = [{"role": "system", "content": SYS_PROMPT}] + EXAMPLES + [{"role": "user", "content": text}]
    result = prompt_llm(messages)
    result['gth'] = data_item['label']
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data", type=str, default='../funny_data/data_raw', help='text_list')
    parser.add_argument("--save_file", type=str, default='../funny_data/data_gen_output/funny_text_only_pred_qwen2.json', help='save file path')
    args = parser.parse_args()

    files = ['val_data.json', 'test_data.json', 'train_data.json']
    dataset = {k: v for file in files for k, v in json.load(open(os.path.join(args.text_data, file))).items()}

    results = json.load(open(args.save_file)) if os.path.exists(args.save_file) else {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_key = {executor.submit(process_text, value): key for key, value in dataset.items() if key not in results or results[key]['logits'] is None}
        for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc='Processing'):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {'logits': None, 'gth': dataset[key]['label']}
                print(f"Error processing {key}: {e}")
            save_results(results, args.save_file)

    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = apply_thresholds(results, thresholds)
    best_threshold = max(f1_scores, key=f1_scores.get)
    results = add_pred_based_on_threshold(results, best_threshold)
    save_results(results, args.save_file)

if __name__ == "__main__":
    main()