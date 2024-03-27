import json

with open('./nycartoon_matching_split_test_text_cls_.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    id, response = line.split(" ", 1)
    data = json.loads(response)
    print(data)