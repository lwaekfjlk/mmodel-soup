import json
import random

# idiom
with open("data_split_output/irfl_idiom_AS_dataset_train.json", 'r') as f:
    idiom_AS_train = json.load(f)
with open("data_split_output/irfl_idiom_R_dataset_train.json", 'r') as f:
    idiom_R_train = json.load(f)
with open("data_split_output/irfl_idiom_U_dataset_train.json", 'r') as f:
    idiom_U_train = json.load(f)

# metaphor
with open("data_split_output/irfl_metaphor_AS_dataset_train.json", 'r') as f:
    metaphor_AS_train = json.load(f)
with open("data_split_output/irfl_metaphor_R_dataset_train.json", 'r') as f:
    metaphor_R_train = json.load(f)
with open("data_split_output/irfl_metaphor_U_dataset_train.json", 'r') as f:
    metaphor_U_train = json.load(f)

# simile
with open("data_split_output/irfl_simile_AS_dataset_train.json", 'r') as f:
    simile_AS_train = json.load(f)
with open("data_split_output/irfl_simile_R_dataset_train.json", 'r') as f:
    simile_R_train = json.load(f)
with open("data_split_output/irfl_simile_U_dataset_train.json", 'r') as f:
    simile_U_train = json.load(f)

def merge_mixed_data(data1, data2):
    # check duplicate keys
    count = 0
    for key, val in data2.items():
        unique_key = f"{key}_{val['text']}"
        if not key in data1.keys():
            data1[unique_key] = val
        else:
            data1[unique_key] = random.choice([data1[unique_key], val])
            count += 1
    print("Number of duplicate keys: ", count)
    return data1

AS_train, R_train, U_train = {}, {}, {}
AS_train = merge_mixed_data(AS_train, idiom_AS_train)
AS_train = merge_mixed_data(AS_train, metaphor_AS_train)
AS_train = merge_mixed_data(AS_train, simile_AS_train)

R_train = merge_mixed_data(R_train, idiom_R_train)
R_train = merge_mixed_data(R_train, metaphor_R_train)
R_train = merge_mixed_data(R_train, simile_R_train)

U_train = merge_mixed_data(U_train, idiom_U_train)
U_train = merge_mixed_data(U_train, metaphor_U_train)
U_train = merge_mixed_data(U_train, simile_U_train)

def get_count_label(data):
    count = 0
    for key, val in data.items():
        if "Figurative" in val['category']:
            count += 1
    return count / len(data), count, len(data)

print("AS: ", get_count_label(AS_train))
print("R: ", get_count_label(R_train))
print("U: ", get_count_label(U_train))
print("Total: ", len(AS_train) + len(R_train) + len(U_train))

with open("data_split_output/irfl_AS_dataset_train.json", 'w') as f:
    json.dump(AS_train, f, indent=4)
with open("data_split_output/irfl_R_dataset_train.json", 'w') as f:
    json.dump(R_train, f, indent=4)
with open("data_split_output/irfl_U_dataset_train.json", 'w') as f:
    json.dump(U_train, f, indent=4)

AS_R_train = merge_mixed_data(AS_train, R_train)
with open("data_split_output/irfl_AS_R_dataset_train.json", 'w') as f:
    json.dump(AS_R_train, f, indent=4)

print("AS + R: ", get_count_label(AS_R_train))

with open("data_raw/irfl_idiom_dataset_train.json", 'r') as f:
    idiom_train = json.load(f)
with open("data_raw/irfl_metaphor_dataset_train.json", 'r') as f:
    metaphor_train = json.load(f)
with open("data_raw/irfl_simile_dataset_train.json", 'r') as f:
    simile_train = json.load(f)

print("Idiom train: ", len(idiom_train))
print("Metaphor train: ", len(metaphor_train))
print("Simile train: ", len(simile_train))

baseline_train = {}
baseline_train = merge_mixed_data(baseline_train, idiom_train)
baseline_train = merge_mixed_data(baseline_train, metaphor_train)
baseline_train = merge_mixed_data(baseline_train, simile_train)

print("Baseline train: ", len(baseline_train))

with open("data_raw/irfl_idiom_dataset_valid.json", 'r') as f:
    idiom_valid = json.load(f)
with open("data_raw/irfl_metaphor_dataset_valid.json", 'r') as f:
    metaphor_valid = json.load(f)
with open("data_raw/irfl_simile_dataset_valid.json", 'r') as f:
    simile_valid = json.load(f)

print("Idiom valid: ", len(idiom_valid))
print("Metaphor valid: ", len(metaphor_valid))
print("Simile valid: ", len(simile_valid))

baseline_valid = {}
baseline_valid = merge_mixed_data(baseline_valid, idiom_valid)
baseline_valid = merge_mixed_data(baseline_valid, metaphor_valid)
baseline_valid = merge_mixed_data(baseline_valid, simile_valid)

print("Baseline valid: ", len(baseline_valid))

with open("data_raw/irfl_idiom_dataset_test.json", 'r') as f:
    idiom_test = json.load(f)
with open("data_raw/irfl_metaphor_dataset_test.json", 'r') as f:
    metaphor_test = json.load(f)
with open("data_raw/irfl_simile_dataset_test.json", 'r') as f:
    simile_test = json.load(f)

print("Idiom test: ", len(idiom_test))
print("Metaphor test: ", len(metaphor_test))
print("Simile test: ", len(simile_test))

baseline_test = {}
baseline_test = merge_mixed_data(baseline_test, idiom_test)
baseline_test = merge_mixed_data(baseline_test, metaphor_test)
baseline_test = merge_mixed_data(baseline_test, simile_test)

print("Baseline test: ", len(baseline_test))

print("baseline train: ", get_count_label(baseline_train))
print("baseline valid: ", get_count_label(baseline_valid))
print("baseline test: ", get_count_label(baseline_test))
print("Total: ", len(baseline_train) + len(baseline_valid) + len(baseline_test))

with open("data_raw/irfl_dataset_train.json", 'w') as f:
    json.dump(baseline_train, f, indent=4)
with open("data_raw/irfl_dataset_valid.json", 'w') as f:
    json.dump(baseline_valid, f, indent=4)
with open("data_raw/irfl_dataset_test.json", 'w') as f:
    json.dump(baseline_test, f, indent=4)