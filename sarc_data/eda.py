import json

with open("data_split_output/sarc_AS_dataset_train.json", "r") as f:
    AS_train = json.load(f)
with open("data_split_output/sarc_R_dataset_train.json", "r") as f:
    R_train = json.load(f)
with open("data_split_output/sarc_U_dataset_train.json", "r") as f:
    U_train = json.load(f)

def get_label_distribution(data):
    count = 0
    for key, val in data.items():
        count += val['label']
    return count / len(data), count, len(data)

print("AS train:", get_label_distribution(AS_train))
print("R train:", get_label_distribution(R_train))
print("U train:", get_label_distribution(U_train))

# upscale R for 0.5 ratio
R_train_upscaled = {}