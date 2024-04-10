import json
from collections import defaultdict
import torch
from datasets import load_dataset

def get_prediction():
    prediction = defaultdict(dict)
    with open('/root/zqi2/backup-mmodel-soup/nycartoon_data/nycartoon_matching_split_train_text_cls_.txt', 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, json_str = line.split(' ', 1)
            data = json.loads(json_str)
            for caption, logits in data.items():
                if 'Yes' in logits.keys() and 'No' in logits.keys():
                    softmax_res = torch.nn.functional.softmax(torch.tensor([logits['Yes'] / 5, logits['No'] / 5]), dim=0)
                    prediction[identifier][caption] = {'yes': softmax_res[0].item(), 'no': softmax_res[1].item()}
    return prediction

def get_score_distribution(prediction):
    score_distriution = defaultdict(list)
    for identifier, captions in prediction.items():
        for caption, scores in captions.items():
            score_distriution[identifier].append(scores['yes'])
    return score_distriution


def select_subset_ids(score_distribution):
    U_ids = []
    AS_ids = []
    R_ids = []
    delta_1 = 0.3 # difference between top1 max and top2 max
    delta_2 = 0.1 # difference between top1 max and top2 max
    for identifier, scores in score_distribution.items():
        scores.sort(reverse=True)
        if len(scores) == 1:
            AS_ids.append(identifier)
        elif scores[0] - scores[1] > delta_1:
            U_ids.append(identifier)
        elif scores[0] - scores[1] < delta_2:
            R_ids.append(identifier)
        else:
            AS_ids.append(identifier)
    return U_ids, R_ids, AS_ids

def construct_subset(ids):
    dataset_name = "jmhessel/newyorker_caption_contest"
    dataset = load_dataset(dataset_name, 'matching')
    subset = dataset["train"].filter(lambda example: example['instance_id'] in ids)
    return subset


if __name__ == "__main__":
    prediction = get_prediction()
    score_distribution = get_score_distribution(prediction)
    U_ids, R_ids, AS_ids = select_subset_ids(score_distribution)
    U_dataset = construct_subset(U_ids)
    R_dataset = construct_subset(R_ids)
    AS_dataset = construct_subset(AS_ids)
    U_dataset.save_to_disk('../../nycartoon_data/results/nycartoon_split_train_U')
    R_dataset.save_to_disk('../../nycartoon_data/results/nycartoon_split_train_R')
    AS_dataset.save_to_disk('../../nycartoon_data/results/nycartoon_split_train_AS')
    print(len(U_dataset))
    print(len(R_dataset))
    print(len(AS_dataset))
    import pdb; pdb.set_trace()
    
