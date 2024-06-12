import json
import ipdb
from datasets import Dataset
from datasets import load_dataset



with open('irfl_idiom_AS_dataset_train.json', 'r') as json_file:
    as_idiom = json.load(json_file)

with open('irfl_idiom_U_dataset_train.json', 'r') as json_file:
    u_idiom = json.load(json_file)

with open('irfl_idiom_R_dataset_train.json', 'r') as json_file:
    r_idiom = json.load(json_file)

with open('irfl_simile_AS_dataset_train.json', 'r') as json_file:
    as_simile = json.load(json_file)

with open('irfl_simile_U_dataset_train.json', 'r') as json_file:
    u_simile = json.load(json_file)

with open('irfl_simile_R_dataset_train.json', 'r') as json_file:
    r_simile = json.load(json_file)

with open('irfl_metaphor_AS_dataset_train.json', 'r') as json_file:
    as_met = json.load(json_file)

with open('irfl_metaphor_U_dataset_train.json', 'r') as json_file:
    u_met = json.load(json_file)

with open('irfl_metaphor_R_dataset_train.json', 'r') as json_file:
    r_met = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_idiom_test.json', 'r') as json_file:
    test_idiom = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_simile_test.json', 'r') as json_file:
    test_simile = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_metaphor_test.json', 'r') as json_file:
    test_metaphor = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_idiom_train.json', 'r') as json_file:
    train_idiom = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_simile_train.json', 'r') as json_file:
    train_simile = json.load(json_file)

with open('/storage/mmodel-soup/irfl_data/data_raw/irfl_metaphor_train.json', 'r') as json_file:
    train_metaphor = json.load(json_file)

with open('irfl_captions.json', 'r') as json_file:
    parsed_results = json.load(json_file)

similes_AS = []
for example in as_simile:
   task = {}
   task['text'] = as_simile[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if as_simile[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif as_simile[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "simile"
   similes_AS.append(task)

# Write the list of dictionaries to the JSON file
with open("AS_similies.json", 'w') as json_file:
    print("saving AS similies")
    json.dump(similes_AS, json_file, indent=4)


similes_R = []
for example in r_simile:
   task = {}
   task['text'] = r_simile[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if r_simile[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif r_simile[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "simile"
   similes_R.append(task)

# Write the list of dictionaries to the JSON file
with open("R_similies.json", 'w') as json_file:
    print("saving similies_R")
    json.dump(similes_R, json_file, indent=4)


similes_U = []
for example in u_simile:
   task = {}
   task['text'] = u_simile[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if u_simile[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif u_simile[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "simile"
   similes_U.append(task)

# Write the list of dictionaries to the JSON file
with open("U_similies.json", 'w') as json_file:
    print("saving similies_U")
    json.dump(similes_U, json_file, indent=4)




idioms_AS = []
for example in as_idiom:
   task = {}
   task['text'] = as_idiom[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if as_idiom[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif as_idiom[example]['category'] == "Figurative":
      task['label'] = 1
   elif as_idiom[example]['category'] == "Figurative+Literal":
      task['label'] = 2
   elif as_idiom[example]['category'] == "Literal":
      task['label'] = 3
   elif as_idiom[example]['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = "idiom"
   idioms_AS.append(task)

with open("AS_idioms.json", 'w') as json_file:
    print("saving idioms_AS")
    json.dump(idioms_AS, json_file, indent=4)

idioms_R = []
for example in r_idiom:
   task = {}
   task['text'] = r_idiom[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if r_idiom[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif r_idiom[example]['category'] == "Figurative":
      task['label'] = 1
   elif r_idiom[example]['category'] == "Figurative+Literal":
      task['label'] = 2
   elif r_idiom[example]['category'] == "Literal":
      task['label'] = 3
   elif r_idiom[example]['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = "idiom"
   idioms_R.append(task)

with open("R_idioms.json", 'w') as json_file:
    print("saving idioms_R")
    json.dump(idioms_R, json_file, indent=4)


idioms_U = []
for example in u_idiom:
   task = {}
   task['text'] = u_idiom[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if u_idiom[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif u_idiom[example]['category'] == "Figurative":
      task['label'] = 1
   elif u_idiom[example]['category'] == "Figurative+Literal":
      task['label'] = 2
   elif u_idiom[example]['category'] == "Literal":
      task['label'] = 3
   elif u_idiom[example]['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = "idiom"
   idioms_U.append(task)

with open("U_idioms.json", 'w') as json_file:
    print("saving idioms_U")
    json.dump(idioms_U, json_file, indent=4)



metaphors_AS = []
for example in as_met:
   task = {}
   task['text'] = as_met[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if as_met[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif as_met[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "metaphor"
   if as_met[example]['category'] == "null":
      print("HELLO")
   metaphors_AS.append(task)


with open("AS_metaphors.json", 'w') as json_file:
    print("saving metaphors_AS")
    json.dump(metaphors_AS, json_file, indent=4) 


metaphors_R = []
for example in r_met:
   task = {}
   task['text'] = r_met[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if r_met[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif r_met[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "metaphor"
   if r_met[example]['category'] == "null":
      print("HELLO")
   metaphors_R.append(task)


with open("R_metaphors.json", 'w') as json_file:
    print("saving metaphors_R")
    json.dump(metaphors_R, json_file, indent=4) 


metaphors_U = []
for example in u_met:
   task = {}
   task['text'] = u_met[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if u_met[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif u_met[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "metaphor"
   if u_met[example]['category'] == "null":
      print("HELLO")
   metaphors_U.append(task)


with open("U_metaphors.json", 'w') as json_file:
    print("saving metaphors_U")
    json.dump(metaphors_U, json_file, indent=4) 


idioms = []
for example in test_idiom:
   task = {}
   task['text'] = test_idiom[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if test_idiom[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif test_idiom[example]['category'] == "Figurative":
      task['label'] = 1
   elif test_idiom[example]['category'] == "Figurative+Literal":
      task['label'] = 2
   elif test_idiom[example]['category'] == "Literal":
      task['label'] = 3
   elif test_idiom[example]['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = "idiom"
   idioms.append(task)

with open("test_idioms.json", 'w') as json_file:
    print("saving idioms")
    json.dump(idioms, json_file, indent=4)

similies = []
for example in test_simile:
   task = {}
   task['text'] = test_simile[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if test_simile[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif test_simile[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "simile"
   similies.append(task)

# Write the list of dictionaries to the JSON file
with open("test_similies.json", 'w') as json_file:
    print("saving similies")
    json.dump(similies, json_file, indent=4)


metaphors = []
for example in test_metaphor:
   task = {}
   task['text'] = test_metaphor[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if test_metaphor[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif test_metaphor[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "metaphor"
   if test_metaphor[example]['category'] == "null":
      print("HELLO")
   metaphors.append(task)

with open("test_metaphors.json", 'w') as json_file:
    print("saving metaphors")
    json.dump(metaphors, json_file, indent=4) 


idioms = []
for example in train_idiom:
   task = {}
   task['text'] = train_idiom[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if train_idiom[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif train_idiom[example]['category'] == "Figurative":
      task['label'] = 1
   elif train_idiom[example]['category'] == "Figurative+Literal":
      task['label'] = 2
   elif train_idiom[example]['category'] == "Literal":
      task['label'] = 3
   elif train_idiom[example]['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = "idiom"
   idioms.append(task)

with open("train_idioms.json", 'w') as json_file:
    print("saving idioms")
    json.dump(idioms, json_file, indent=4)

similies = []
for example in train_simile:
   task = {}
   task['text'] = train_simile[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if train_simile[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif train_simile[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "simile"
   similies.append(task)

# Write the list of dictionaries to the JSON file
with open("train_similies.json", 'w') as json_file:
    print("saving similies")
    json.dump(similies, json_file, indent=4)


metaphors = []
for example in train_metaphor:
   task = {}
   task['text'] = train_metaphor[example]['text']
   task['image_id'] = example
   img_key = example + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if train_metaphor[example]['category'] == "Partial Literal":
      task['label'] = 0
   elif train_metaphor[example]['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = "metaphor"
   if train_metaphor[example]['category'] == "null":
      print("HELLO")
   metaphors.append(task)

with open("train_metaphors.json", 'w') as json_file:
    print("saving metaphors")
    json.dump(metaphors, json_file, indent=4) 
