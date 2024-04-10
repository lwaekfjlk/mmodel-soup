import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import json
import json
import ast
from collections import defaultdict

def read_vision_and_text_labels():
    text_label_dict = defaultdict(dict)
    vision_label_dict = defaultdict(dict)

    with open("../../sarc_data/intermediate_data/sarc_text_label_logits_eval.txt", 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, json_str = line.split(' ', 1)
            data = json.loads(json_str)
            text_label_dict[identifier] = {'yes': -float('inf'), 'no': -float('inf')}
            if 'Yes' in data.keys() and 'No' in data.keys():
                text_label_dict[identifier]['yes'] = data['Yes']
                text_label_dict[identifier]['no'] = data['No']
                text_label_dict[identifier]['pred'] = 1 if data['Yes'] > data['No'] else 0

    with open("../../sarc_data/intermediate_data/sarc_vision_label_logits.txt", 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, list_str = line.split(' ', 1)
            identifier = identifier.split('.')[0]
            data = ast.literal_eval(list_str.strip().replace('\"', ''))
            vision_label_dict[identifier]['yes'] = data[0]
            vision_label_dict[identifier]['no'] = data[1]
            vision_label_dict[identifier]['pred'] = 1 if data[0] > data[1] else 0

    print(len(text_label_dict))
    print(len(vision_label_dict))
    return text_label_dict, vision_label_dict


def read_groundtruth_labels():
    gth_label_dict = {}
    dataset = {}
    with open("../../sarc_data/intermediate_data/train_filtered.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        data_list = ast.literal_eval(line)
        img_id = data_list[0]
        label = data_list[-1]
        gth_label_dict[img_id] = label
        dataset[img_id] = {'text': data_list[1], 'label': label}
    return gth_label_dict, dataset

def select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict):
    R_ids = []
    AS_ids = []
    U_ids = []
    for id, gth_label in gth_label_dict.items():
        try:
            text_label = text_label_dict[id]['pred']
            vision_label = vision_label_dict[id]['pred']
            if text_label == vision_label and text_label == gth_label:
                R_ids.append(id)
            if text_label == vision_label and text_label != gth_label:
                AS_ids.append(id)
            if text_label != vision_label:
                U_ids.append(id)
        except:
            if 'pred' not in text_label_dict[id].keys():
                print('no text_label')
            if 'pred' not in vision_label_dict[id].keys():
                print('no vision_label')
            continue
    return R_ids, AS_ids, U_ids


def construct_subset(ids, dataset):
    subset = {}
    for id in ids:
        subset[id] = dataset[id]
    return subset



if __name__ == "__main__":
    text_label_dict, vision_label_dict = read_vision_and_text_labels()
    gth_label_dict, dataset = read_groundtruth_labels()
    R_ids, AS_ids, U_ids = select_subset_ids(text_label_dict, vision_label_dict, gth_label_dict)
    R_dataset = construct_subset(R_ids, dataset)
    with open('../../sarc_data/results/sarc_R_dataset_train.json', 'w') as f:
        json.dump(R_dataset, f)
    AS_dataset = construct_subset(AS_ids, dataset)
    with open('../../sarc_data/results/sarc_AS_dataset_train.json', 'w') as f:
        json.dump(AS_dataset, f)
    U_dataset = construct_subset(U_ids, dataset)
    with open('../../sarc_data/results/sarc_U_dataset_train.json', 'w') as f:
        json.dump(U_dataset, f)

    print(len(R_ids))
    print(len(AS_ids))
    print(len(U_ids))