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

litellm.set_verbose = False

SYS_PROMPT = (
    "Please analyze the text provided below for sarcasm."
    "If you think the text includes exaggerated description or includes strong emotion or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think the text is neutral or its true meaning is not different from its original one, please answer 'No'."
    "Please make sure that your answer is based on the text itself, not on the context or your personal knowledge."
    "There are only two options: 'Yes' or 'No'."
    "If you are not sure, please provide your best guess and do not say that you are not sure."
    "You should only make Yes judgement when you are very sure that the text is sarcastic."
)

EXAMPLES = [
    {"role": "user", "content": "TEXT: Yes yes it is! In Prison!!"},
    {"role": "assistant", "content": "Yes. It expresses the speaker's strong emotion about the situation which indicates that the speaker is sarcastic."},
    {"role": "user", "content": "TEXT: And then and then you clicked it again, she's dressed. She is a business woman, she is walking down the street and oh oh oh she's naked."},
    {"role": "assistant", "content": "No. It is a neutral statement."}
]

def process_text(data_item: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    text = f'TEXT: {data_item["utterance"]}'
    messages = [{"role": "system", "content": SYS_PROMPT}] + EXAMPLES + [{"role": "user", "content": text}]
    result = prompt_llm(messages)
    result['gth'] = int(data_item['sarcasm'])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data", type=str, default='../mustard_data/data_raw', help='text_list')
    parser.add_argument("--save_file", type=str, default='../mustard_data/data_gen_output/mustard_text_only_pred_qwen2.json', help='save file path')
    args = parser.parse_args()

    files = ['mustard_dataset_train.json', 'mustard_dataset_test.json']
    dataset = {k: v for file in files for k, v in json.load(open(os.path.join(args.text_data, file))).items()}

    results = json.load(open(args.save_file)) if os.path.exists(args.save_file) else {}

    multi_process_run(process_text, results, dataset, args.max_workers, args.save_file)

    thresholds = np.arange(0, 1.0, 0.01)
    f1_scores = apply_thresholds(results, thresholds)
    best_threshold = max(f1_scores, key=f1_scores.get)
    print(f"Best Threshold: {best_threshold}, Best F1 Score: {f1_scores[best_threshold]}")

    results = add_pred_based_on_threshold(results, best_threshold)
    save_results(results, args.save_file)

if __name__ == "__main__":
    main()
