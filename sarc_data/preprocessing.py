import ast
import os

with open("raw_data/valid.txt", 'r') as f:
    input_str = f.read()
    lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
    lists = [ast.literal_eval(line) for line in lines]
    
image_root = "raw_data/image_data"
new_lists = []
for image_id, sentence, label, _ in lists:
    if os.path.exists(os.path.join(image_root,'%s.jpg'%image_id)):
        new_lists.append([image_id, sentence, label])

with open("intermediate_data/valid_filtered.txt", 'w') as f:
    for line in new_lists:
        f.write(str(line) + '\n')
