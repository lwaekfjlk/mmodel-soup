import json
import argparse
from typing import Dict, List
import concurrent.futures
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import Counter
from utils import (
    prompt_llm,
    save_results,
    get_prediction,
    multi_process_run,
    load_dataset,
    load_ids,
    calculate_f1
)

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
    parser.add_argument("--max_workers", type=int, default=32, help='max workers')
    args = parser.parse_args()

    files = ['sarc_dataset_test.json', 'sarc_dataset_train.json', 'sarc_dataset_val.json']
    train_ids = load_ids('../sarc_data/data_raw/sarc_dataset_train.json')
    val_ids = load_ids('../sarc_data/data_raw/sarc_dataset_val.json')
    test_ids = load_ids('../sarc_data/data_raw/sarc_dataset_test.json')

    dataset = load_dataset(files, args.text_data)

    results = json.load(open(args.save_file)) if os.path.exists(args.save_file) else {}

    #multi_process_run(process_text, results, dataset, args.max_workers, args.save_file)

    # Process results and calculate F1 scores
    train_results = get_prediction({k: v for k, v in results.items() if k in train_ids}, 0.2, split='train')
    val_results = get_prediction({k: v for k, v in results.items() if k in val_ids}, 0, split='val')
    test_results = get_prediction({k: v for k, v in results.items() if k in test_ids}, 0, split='test')

    print(f"Train F1 Score: {calculate_f1(train_results, train_ids)}")
    print(f"Validation F1 Score: {calculate_f1(val_results, val_ids)}")
    print(f"Test F1 Score: {calculate_f1(test_results, test_ids)}")

    # Merge results and save
    results = {**train_results, **val_results, **test_results}
    save_results(results, args.save_file)

if __name__ == "__main__":
    main()
