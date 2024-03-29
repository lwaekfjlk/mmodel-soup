import os
import json
import csv
import re
from typing import List, Tuple, Dict, Union
from sklearn.metrics import f1_score, precision_score, recall_score

def eval_metric(results: Dict[str, Dict[str, str]], labels: Dict[str, bool], prediction_key: str='prediction') -> Dict[str, float]:
    """Compute evaluation metrics for predictions compared to ground truth."""
    
    shared_keys = set(results.keys()) & set(labels.keys())
    predictions = [results[key][prediction_key] for key in shared_keys]
    ground_truth = [labels[key] for key in shared_keys]

    acc = sum([1 if p == g else 0 for p, g in zip(predictions, ground_truth)]) / len(ground_truth)

    return {
        'acc': acc, 
        'f1': f1_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions),
        'recall': recall_score(ground_truth, predictions)
    }

def load_dataset(args: object) -> Tuple[Dict[str, Dict[str, Union[int, str]]], Dict[str, int]]:
    """Load a dataset and its corresponding labels."""

    if args.speaker_independent: 
        train_dataset_path = os.path.join(args.dataset_directory, 'sarcasm_data_speaker_independent_train.json')
        test_dataset_path = os.path.join(args.dataset_directory, 'sarcasm_data_speaker_independent_test.json')
    else:
        train_dataset_path = os.path.join(args.dataset_directory, 'sarcasm_data.json')
        test_dataset_path = os.path.join(args.dataset_directory, 'sarcasm_data.json')

    with open(train_dataset_path, 'r') as f:
        train_dataset = json.load(f)
    
    with open(test_dataset_path, 'r') as f:
        test_dataset = json.load(f)
    
    if args.use_subpart:
        filename = f'{args.modality1}_{args.modality2}_{args.agree_or_not}.json' if args.multimodal else f'{args.modality1}_{args.agree_or_not}.json'
        subpart_path = os.path.join(args.subpart_directory, filename)
        
        with open(subpart_path, 'r') as f:
            subpart_ids = json.load(f)
        
        subpart_train_dataset = {idx: train_dataset[idx] for idx in subpart_ids.keys()}
        subpart_test_dataset = {idx: test_dataset[idx] for idx in subpart_ids.keys()}
        subpart_train_labels = {key: data['sarcasm'] for key, data in train_dataset.items()}
        subpart_test_labels = {key: data['sarcasm'] for key, data in test_dataset.items()}
        return subpart_train_dataset, subpart_test_dataset, subpart_train_labels, subpart_test_labels
    
    return train_dataset, test_dataset, {key: data['sarcasm'] for key, data in train_dataset.items()}, {key: data['sarcasm'] for key, data in test_dataset.items()}

def load_audio_emotion(args: object) -> Dict[str, str]:
    """Load audio information from a CSV."""
    
    audio_info_path = os.path.join(args.audio_info_directory, args.audio_info_filename)
    audio_info = {}
    
    with open(audio_info_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            audio_info[row[0]] = row[-1]
    
    return audio_info

def load_audio_description(args: object) -> Dict[str, str]:
    # load audio description from json
    audio_description_path = os.path.join(args.audio_description_directory_from_video_llama, args.audio_description_filename_from_video_llama)
    audio_description = {}
    with open(audio_description_path, 'r') as f:
        audio_description = json.load(f)
    return audio_description


def load_vision_description(args: object) -> Dict[str, str]:
    # load vision description from json
    vision_description_path = os.path.join(args.vision_description_directory_from_video_llama, args.vision_description_filename_from_video_llama)
    vision_description = {}
    with open(vision_description_path, 'r') as f:
        vision_description = json.load(f)
    return vision_description



def load_face_emotion(args: object) -> Dict[str, List[str]]:
    """Load vision-related information from a CSV."""
    
    vision_info_path = os.path.join(args.vision_info_directory, args.vision_info_filename)
    vision_info = {}
    
    with open(vision_info_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vision_info[row[0]] = row[1:]
    
    return vision_info


def write_output(args: object, results: Dict[str, Dict[str, str]]) -> None:
    """Write results to a JSON file."""
    from datetime import datetime

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YearMonthDayHourMinuteSecond

    if args.multimodal:
        filename = f'{args.modality1}_{args.modality2}_{args.agree_or_not}_output_num_{len(results)}_{current_time}.json'
    else:
        filename = f'{args.modality1}_{args.agree_or_not}_output_num_{len(results)}_{current_time}.json'

    output_path = os.path.join(args.output_directory, filename)
    with open(output_path, 'w') as f:
        json.dump(results, f)

def filter_invalid_ans(args: object, ans: str) -> (bool, int):
    """
    Extracts judgement and confidence from an answer string.
    - Returns True/False based on the presence of 'Yes'/'No'.
    - Extracts confidence number if present.
    """
    predicted_number = check_numbers(ans)
    try:
        if predicted_number > 0:
            judgement = True
        elif predicted_number < 0:
            judgement = False
        else:
            judgement = None
    except:
        judgement = None

    if args.predict_confidence:
        confidence = check_numbers(ans)
        return judgement, predicted_number, confidence
    else:
        return judgement, predicted_number, None


def check_numbers(string: str) -> float:
    """
    Extracts the first real number between -5 and 5 from the string.
    - Returns the number if found.
    - Returns None otherwise.
    """
    # This regex matches real numbers between -5 and 5
    match = re.search(r"(-?5(?![\.\d])|-?\d(\.\d+)?)", string)
    
    return float(match.group()) if match else None