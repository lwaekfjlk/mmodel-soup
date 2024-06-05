import json

with open("irfl_idiom_dataset_test.json", "r") as f:
    irfl_idiom_dataset_test = json.load(f)
with open("irfl_idiom_dataset_train.json", "r") as f:
    irfl_idiom_dataset_train = json.load(f)
with open("irfl_idiom_dataset_valid.json", "r") as f:
    irfl_idiom_dataset_valid = json.load(f)

with open("irfl_metaphor_dataset_train.json", "r") as f:
    irfl_metaphor_dataset_train = json.load(f)
with open("irfl_metaphor_dataset_valid.json", "r") as f:
    irfl_metaphor_dataset_valid = json.load(f)
with open("irfl_metaphor_dataset_test.json", "r") as f:
    irfl_metaphor_dataset_test = json.load(f)

with open("irfl_simile_dataset_train.json", "r") as f:
    irfl_simile_dataset_train = json.load(f)
with open("irfl_simile_dataset_valid.json", "r") as f:
    irfl_simile_dataset_valid = json.load(f)
with open("irfl_simile_dataset_test.json", "r") as f:
    irfl_simile_dataset_test = json.load(f)

idiom_dataset, metaphor_dataset, simile_dataset = {}, {}, {}
idiom_dataset.update(irfl_idiom_dataset_train)
idiom_dataset.update(irfl_idiom_dataset_valid)
idiom_dataset.update(irfl_idiom_dataset_test)
metaphor_dataset.update(irfl_metaphor_dataset_train)
metaphor_dataset.update(irfl_metaphor_dataset_valid)
metaphor_dataset.update(irfl_metaphor_dataset_test)
simile_dataset.update(irfl_simile_dataset_train)
simile_dataset.update(irfl_simile_dataset_valid)
simile_dataset.update(irfl_simile_dataset_test)

idiom_dataset.update(simile_dataset)
dataset_train = idiom_dataset
# evenly split metaphor dataset into valid and test
dataset_valid, dataset_test = {}, {}
for i, (k, v) in enumerate(metaphor_dataset.items()):
    if i % 2 == 0:
        dataset_valid[k] = v
    else:
        dataset_test[k] = v

print(len(dataset_train), len(dataset_valid), len(dataset_test))
with open("irfl_dataset_train.json", "w") as f:
    json.dump(dataset_train, f)

with open("irfl_dataset_valid.json", "w") as f:
    json.dump(dataset_valid, f)

with open("irfl_dataset_test.json", "w") as f:
    json.dump(dataset_test, f)