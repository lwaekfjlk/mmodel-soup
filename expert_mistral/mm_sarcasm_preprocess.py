# Write sarcasm data to TSV file
import ipdb
import ast
import os
import csv
from datasets import load_dataset


with open("root/bak/CogVLM/sarc_data/text_data/test.txt", 'r') as f:
    input_str = f.read()
    lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
    lists = [ast.literal_eval(line) for line in lines]

image_root = "root/bak/CogVLM/sarc_data/image_data"
new_lists = []
for image_id, sentence, label, ground in lists:
    #print(os.path.join(image_root,'%s.jpg'%image_id))
    #ipdb.set_trace()
    if os.path.exists(os.path.join(image_root,'%s.jpg'%image_id)):
        new_lists.append([image_id, sentence, label])
        #print(image_id)
print(len(new_lists))

with open("root/bak/CogVLM/sarc_data/text_data/text_data/test_filtered.txt", 'w') as f:
    for line in new_lists:
        f.write(str(line) + '\n')

with open('root/bak/CogVLM/sarc_data/text_data/text_data/test_filtered.txt', 'r') as f:
    input_str = f.read()
    lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
    lists = [ast.literal_eval(line) for line in lines]