import json
import argparse
from typing import Dict, List
import concurrent.futures
import os
import numpy as np
from tqdm import tqdm
from utils import prompt_llm, save_results, apply_thresholds, add_pred_based_on_threshold

SYS_PROMPT = (
    "Please analyze the text provided below for sarcasm."
    "If you think the text includes exaggerated description or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think the text is neutral or its true meaning is not different from its original one, please answer 'No'."
    "Please make sure that your answer is based on the text itself, not on the context or your personal knowledge."
    "There are only two options: 'Yes' or 'No'."
    "If you are not sure, please provide your best guess and do not say that you are not sure."
)

EXAMPLES = [
    {"role": "user", "content": "TEXT: because lunch is more interesting than job and even tasty..."},
    {"role": "assistant", "content": "Yes. It expresses the speaker's preference for lunch over job by using the word 'tasty'."},
    {"role": "user", "content": "TEXT: gameday ready # alabamacrimsontide emoji_53'"},
    {"role": "assistant", "content": "No. It is a neutral statement."}
]


def process_text(data_item: Dict[str, str]) -> Dict[str, int]:
    text = 'TEXT: ' + data_item['text']
    messages = [{"role": "system", "content": SYS_PROMPT}] + EXAMPLES + [{"role": "user", "content": text}]
    result = prompt_llm(messages)
    result['gth'] = data_item['label']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data", type=str, default='../sarc_data/data_raw', help='text_list')
    parser.add_argument("--save_file", type=str, default='../sarc_data/data_gen_output/sarc_text_only_pred_qwen2.json', help='save file path')
    args = parser.parse_args()

    files = ['sarc_dataset_test.json', 'sarc_dataset_train.json', 'sarc_dataset_val.json']
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

    thresholds = np.arange(0, 1.0, 0.01)
    f1_scores = apply_thresholds(results, thresholds)
    best_threshold = max(f1_scores, key=f1_scores.get)
    print(f"Best Threshold: {best_threshold}, Best F1 Score: {f1_scores[best_threshold]}")

    results = add_pred_based_on_threshold(results, best_threshold)
    save_results(results, args.save_file)

if __name__ == "__main__":
    main()
