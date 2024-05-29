import json

with open("nycartoon_matching_split_train_text_label_logits.txt" , 'r') as f:
    lines = f.readlines()
image_id, json_str = lines[0].split(" ", 1)
print(json_str)

final_data = {}
all_image_ids = []
yes_misses, no_misses = 0, 0
for line in lines:
    image_id, json_str = line.split(" ", 1)
    entity = json.loads(json_str)
    all_image_ids.append(image_id)
    for k, v in entity.items():
        if not 'yes' in v:
            yes_misses += 1
        if not 'no' in v:
            no_misses += 1
    
print(len(all_image_ids))
print(len(set(all_image_ids)))
print(yes_misses, no_misses)

final_data = {}
for line in lines:
    image_id, json_str = line.split(" ", 1)
    data = json.loads(json_str)
    all_image_ids.append(image_id)
    final_data[image_id] = {}
    for text in data:
        final_data[image_id][text] = {}
        final_data[image_id][text]['logits'] = {
            'Yes': data[text]['yes'] if 'yes' in data[text] else -float("inf"),
            'No': data[text]['no'] if 'no' in data[text] else -float("inf")
        }
        final_data[image_id][text]['pred'] = 0
    # find the text with the maximum difference
    max_diff = -float("inf")
    max_diff_text = None
    for text in data:
        diff = final_data[image_id][text]['logits']['yes'] - final_data[image_id][text]['logits']['no']
        if diff >= max_diff:
            max_diff = diff
            max_diff_text = text
    final_data[image_id][max_diff_text]['pred'] = 1
    
# validate the data structure
for image_id in final_data:
    for text in final_data[image_id]:
        assert final_data[image_id][text]['pred'] == 0 or final_data[image_id][text]['pred'] == 1
    for text in final_data[image_id]:
        assert 'yes' in final_data[image_id][text]['logits'] and 'no' in final_data[image_id][text]['logits']
        
with open("nycartoon_pred.json", 'w') as f:
    json.dump(final_data, f, indent=4)