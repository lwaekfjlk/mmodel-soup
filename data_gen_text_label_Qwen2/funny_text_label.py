import json
import argparse
import os
from typing import List, Dict
from sklearn.metrics import f1_score
from utils import (
    prompt_llm,
    save_results,
    get_prediction,
    multi_process_run,
    load_dataset,
    load_ids,
    calculate_f1
)

# System prompt for the language model
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
    """Process a single text item to determine if it is humorous."""
    text = 'TEXT: ' + data_item['punchline_sentence']
    messages = [{"role": "system", "content": SYS_PROMPT}] + EXAMPLES + [{"role": "user", "content": text}]
    result = prompt_llm(messages)
    result['gth'] = data_item['label']
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data", type=str, default='../urfunny_data/data_raw', help='Text data directory')
    parser.add_argument("--save_file", type=str, default='../urfunny_data/data_gen_output/urfunny_text_only_pred_qwen2.json', help='Save file path')
    parser.add_argument("--max_workers", type=int, default=64, help='Maximum number of workers for parallel processing')
    args = parser.parse_args()

    files = ['urfunny_dataset_val.json', 'urfunny_dataset_test.json', 'urfunny_dataset_train.json']
    train_ids = load_ids('../urfunny_data/data_raw/urfunny_dataset_train.json')
    val_ids = load_ids('../urfunny_data/data_raw/urfunny_dataset_val.json')
    test_ids = load_ids('../urfunny_data/data_raw/urfunny_dataset_test.json')

    dataset = load_dataset(files, args.text_data)

    results = json.load(open(args.save_file)) if os.path.exists(args.save_file) else {}

    # Uncomment the line below to enable multi-process processing of text data
    # multi_process_run(process_text, results, dataset, args.max_workers, args.save_file)

    # Process results and calculate F1 scores
    train_results = get_prediction({k: v for k, v in results.items() if k in train_ids}, 0.2)
    val_results = get_prediction({k: v for k, v in results.items() if k in val_ids}, 0)
    test_results = get_prediction({k: v for k, v in results.items() if k in test_ids}, 0)

    print(f"Train F1 Score: {calculate_f1(train_results, train_ids)}")
    print(f"Validation F1 Score: {calculate_f1(val_results, val_ids)}")
    print(f"Test F1 Score: {calculate_f1(test_results, test_ids)}")

    # Merge results and save
    results = {**train_results, **val_results, **test_results}
    save_results(results, args.save_file)

if __name__ == "__main__":
    main()
