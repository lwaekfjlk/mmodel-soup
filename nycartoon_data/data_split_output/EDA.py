import json
with open("nycartoon_AS_dataset_train.json", 'r') as f:
    raw_dataset = json.load(f)

lens = []
for id, data in raw_dataset.items():
    image_id, _ = id.split('_')
    caption = data['caption']
    question = data['questions'][0]
    text = f"{question} {caption}"
    lens.append(len(text))

print(len([i for i in lens if i > 150]))
print(len(lens))