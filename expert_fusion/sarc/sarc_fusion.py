import jsonlines
import os
from collections import defaultdict

def build_dataset(file_dir):
    subset_name = ['AS', 'R', 'U']
    dataset = defaultdict(list)
    for name in subset_name:
        # read jsonlines
        with jsonlines.open(os.path.join(file_dir, f'sarc_{name}_vision_text_label.jsonl'), 'r') as f:
            for line in f:
                dataset[name].append(line)
    return dataset


def addition_fusion(dataset):
    import pdb; pdb.set_trace()
    return


if __name__ == "__main__":
    file_dir = '../../sarc_data/intermediate_data/ALBEF_RUS_outputs'
    dataset = build_dataset(file_dir)
    acc = addition_fusion(dataset)
