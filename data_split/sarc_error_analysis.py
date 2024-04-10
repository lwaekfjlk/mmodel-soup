import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import json
import json
import ast
from collections import defaultdict

def visualize(image_dir, image_id, text, pred, label, text_label_dict, image_label_dict):
    print("Image ID: ", image_id)
    print("Text: ", text)
    print("Prediction: ", pred)
    print("Label: ", label)
    # print(text_label_dict)
    # import pdb; pdb.set_trace()
    try:
        print("Text Label Dict: ", text_label_dict[image_id])
        if text_label_dict[image_id]['yes'] > text_label_dict[image_id]['no']:
            print("Text Predict: 1")
        else:
            print("Text Predict: 0")
    except KeyError:
        print("No label for this text")
    try:
        print("Image Label Dict: ", image_label_dict[image_id])
        if image_label_dict[image_id][0] > image_label_dict[image_id][1]:
            print("Image Predict: 1")
        else:
            print("Image Predict: 0")
    except KeyError:
        print("No label for this image")
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
    # Display the image
    img = Image.open(image_path)
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return

if __name__ == "__main__":
    text_label_dict = defaultdict(dict)
    vision_label_dict = defaultdict(dict)

    with open("../sarc_data/intermediate_data/sarc_text_label_logits_eval.txt", 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, json_str = line.split(' ', 1)
            data = json.loads(json_str)
            text_label_dict[identifier] = {'yes': -float('inf'), 'no': -float('inf')}
            if 'Yes' in data.keys() and 'No' in data.keys():
                text_label_dict[identifier]['yes'] = data['Yes']
                text_label_dict[identifier]['no'] = data['No']
                text_label_dict[identifier]['pred'] = 1 if data['Yes'] > data['No'] else 0

    with open("../sarc_data/intermediate_data/sarc_vision_label_logits.txt", 'r') as f:
        reader = f.readlines()
        for line in reader:
            identifier, list_str = line.split(' ', 1)
            identifier = identifier.split('.')[0]
            data = ast.literal_eval(list_str.strip().replace('\"', ''))
            vision_label_dict[identifier]['yes'] = data[0]
            vision_label_dict[identifier]['no'] = data[1]
            vision_label_dict[identifier]['pred'] = 1 if data[0] > data[1] else 0

    print(len(text_label_dict))
    print(len(vision_label_dict))


    with open("../sarc_data/results/sarc_vision_text_label.jsonl", 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    R = {'total_num': 0, 'correct_num': 0}
    AS = {'total_num': 0, 'correct_num': 0}
    U = {'total_num': 0, 'correct_num': 0}
    DS = {'total_num': 0, 'correct_num': 0}

    for line in lines:
        data = json.loads(line)
        image_id = data["image_id"]
        try:
            text_label = text_label_dict[image_id]['pred']
            vision_label = vision_label_dict[image_id]['pred']
        except:
            print('no text_label or vision_label')
            continue
        gth_label = data["target"]
        pred = data["pred"]

        print(text_label, vision_label, gth_label, pred)
        if text_label == vision_label and text_label == gth_label:
            R['total_num'] += 1
            if pred == gth_label:
                R['correct_num'] += 1
        
        if text_label == vision_label and text_label != gth_label:
            AS['total_num'] += 1
            if pred == gth_label:
                AS['correct_num'] += 1
        
        if text_label != vision_label:
            U['total_num'] += 1
            if pred == gth_label:
                U['correct_num'] += 1
    
    print(R)
    print(AS)
    print(U)
    print("R accuracy: ", R['correct_num']/R['total_num'])
    print("AS accuracy: ", AS['correct_num']/AS['total_num'])
    print("U accuracy: ", U['correct_num']/U['total_num'])
    import pdb; pdb.set_trace()


    # visualize the first 5 examples
    count = 0
    for line in lines:
        data = json.loads(line)
        if data["pred"] != data["target"]:
            visualize(
                "raw_data/image_data", 
                data["image_id"], 
                data["text"], 
                data["pred"], 
                data["target"], 
                text_label_dict, 
                vision_label_dict
            )
            count += 1
            if count == 50:
                break