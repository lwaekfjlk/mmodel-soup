import json
import ipdb
from datasets import Dataset
from datasets import load_dataset

# Read the text file


IRFL_images = load_dataset("lampent/IRFL", data_files='IRFL_images.zip')['train']
# IRFL dataset of figurative phrase-image pairs (10k+ images)
IRFL_similes_dataset = load_dataset("lampent/IRFL", 'similes-dataset')['dataset']
IRFL_metaphors_dataset = load_dataset("lampent/IRFL", 'metaphors-dataset')['dataset']
IRFL_idioms_dataset = load_dataset("lampent/IRFL", 'idioms-dataset')['dataset']
with open('irfl_captions.json', 'r') as json_file:
    parsed_results = json.load(json_file)
"""
idioms = []
for example in IRFL_idioms_dataset:
   task = {}
   task['text'] = example['phrase']
   task['image_id'] = example['uuid']
   img_key = example['uuid'] + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if example['category'] == "Partial Literal":
      task['label'] = 0
   elif example['category'] == "Figurative":
      task['label'] = 1
   elif example['category'] == "Figurative+Literal":
      task['label'] = 2
   elif example['category'] == "Literal":
      task['label'] = 3
   elif example['category'] == "No Category":
      task['label'] = 4
   else:
      continue
   task['type'] = example['figurative_type']
   idioms.append(task)


with open("mistral_idioms.json", 'w') as json_file:
    print("saving idioms")
    json.dump(idioms, json_file, indent=4)
 """

#image = get_image('23671399455448973596548783550705533718080580071745863224510509451054839685279')
#image.save("example_image.jpeg")

similies = []
for example in IRFL_similes_dataset:
   task = {}
   task['text'] = example['phrase']
   task['image_id'] = example['uuid']
   img_key = example['uuid'] + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if example['category'] == "Partial Literal":
      task['label'] = 0
   elif example['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = example['figurative_type']
   similies.append(task)


# Write the list of dictionaries to the JSON file
with open("mistral_similies.json", 'w') as json_file:
    print("saving similies")
    json.dump(similies, json_file, indent=4)


metaphors = []
for example in IRFL_metaphors_dataset:
   task = {}
   task['text'] = example['phrase']
   task['image_id'] = example['uuid']
   img_key = example['uuid'] + ".jpeg"
   task['caption'] = parsed_results[img_key]
   if example['category'] == "Partial Literal":
      task['label'] = 0
   elif example['category'] == "Figurative":
      task['label'] = 1
   else:
      continue
   task['type'] = example['figurative_type']
   if example['category'] == "null":
      print("HELLO")
   similies.append(task)


with open("mistal_metaphors.json", 'w') as json_file:
    print("saving metaphors")
    json.dump(similies, json_file, indent=4) 
