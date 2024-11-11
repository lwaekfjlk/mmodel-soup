import json
from collections import Counter
from typing import Dict, List

import jsonlines
from sklearn.metrics import f1_score


def apply_thresholds(results: Dict, thresholds: List[float]) -> Dict[float, float]:
    return {
        threshold: f1_score(
            [value["gth"] for value in results.values() if value["gth"] is not None],
            [
                1 if value["logits"]["Yes"] >= threshold else 0
                for value in results.values()
                if value["gth"] is not None
            ],
        )
        for threshold in thresholds
    }


def add_pred_based_on_threshold(results: Dict, threshold: float) -> Dict:
    for key, value in results.items():
        if value["logits"] and value["gth"] is not None:
            value["pred"] = 1 if value["logits"]["Yes"] >= threshold else 0
        else:
            value["pred"] = None
    return {k: v for k, v in results.items() if v["gth"] is not None}


def save_results_to_json(results, file_path):
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


def get_prediction(results, balance_lower_bound):
    pred_counter = Counter()
    for result in results.values():
        result["pred"] = 1 if result["logits"]["Yes"] > result["logits"]["No"] else 0
        pred_counter[result["pred"]] += 1

    pred_ratio = pred_counter[1] / sum(pred_counter.values())
    if pred_ratio < balance_lower_bound:
        results = select_top_percent_as_one(results, balance_lower_bound)
    elif pred_ratio > 1 - balance_lower_bound:
        results = select_top_percent_as_one(results, 1 - balance_lower_bound)

    print(Counter(result["pred"] for result in results.values()))
    return results


def select_top_percent_as_one(results, percentage):
    sorted_logits = sorted(
        results.items(), key=lambda x: x[1]["logits"]["Yes"], reverse=True
    )
    threshold_index = int(len(sorted_logits) * percentage)

    for i, (image_id, _) in enumerate(sorted_logits):
        results[image_id]["pred"] = 1 if i < threshold_index else 0
    return results


def mask_uncertain_predictions(results, threshold):
    train_count = 0
    for value in results.values():
        if value["logits"] and value["gth"] is not None:
            prob_diff = abs(value["logits"]["Yes"] - value["logits"]["No"])
            should_include_in_train = prob_diff >= threshold
            value["should_include_in_train"] = should_include_in_train
            train_count += should_include_in_train
    print(f"Train count: {train_count}")
    return results


def process_data(file_path, ids, split):
    with jsonlines.open(file_path) as reader:
        labels = list(reader)

    dataset = {}
    for data in labels:
        if data["image_id"] in ids:
            dataset[data["image_id"]] = {
                "logits": data["logits"],
                "gth": data.get("gth"),
            }
    return dataset


def compute_f1(results):
    preds = [value["pred"] for value in results.values() if value["gth"] is not None]
    gths = [value["gth"] for value in results.values() if value["gth"] is not None]
    return f1_score(gths, preds)


if __name__ == "__main__":
    dataset_name = "urfunny"

    with open(
        f"../{dataset_name}_data/data_raw/{dataset_name}_dataset_train.json"
    ) as f:
        train_ids = list(json.load(f).keys())

    with open(f"../{dataset_name}_data/data_raw/{dataset_name}_dataset_val.json") as f:
        val_ids = list(json.load(f).keys())

    with open(f"../{dataset_name}_data/data_raw/{dataset_name}_dataset_test.json") as f:
        test_ids = list(json.load(f).keys())

    train_results = process_data(
        f"../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.jsonl",
        train_ids,
        "train",
    )

    train_results = get_prediction(train_results, 0.0)
    print(f"Train F1 score: {compute_f1(train_results)}")

    other_results = process_data(
        f"../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2.jsonl",
        val_ids + test_ids,
        "other",
    )

    other_results = get_prediction(other_results, 0.0)
    print(f"Other F1 score: {compute_f1(other_results)}")

    results = {**train_results, **other_results}
    save_results_to_json(
        results,
        f"../{dataset_name}_data/data_gen_output/{dataset_name}_image_only_pred_cogvlm2_for_fuser.json",
    )
