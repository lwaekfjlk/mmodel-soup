import json
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import time
from collections import defaultdict
from nycartoon_utils import prompt_construct

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def preprocess(dataset):
    messages = defaultdict(lambda: defaultdict(list))
    for data in dataset:
        instance_id = data['instance_id']

        scene = data['image_location']
        entities = [entity.split('/')[-1] for entity in data['entities']]
        entities = ', '.join(entities)
        image_description = data['image_description']
        image_uncanny_description = data['image_uncanny_description']
        caption_choices = data['caption_choices']

        pos_caption = caption_choices['ABCDE'.index(data['label'])]
        neg_captions = [caption for caption in caption_choices if caption != pos_caption]

        choices = [pos_caption] + neg_captions

        for caption in choices:
            messages[instance_id][caption] = prompt_construct(
                scene, 
                entities, 
                image_description, 
                image_uncanny_description, 
                caption
            )
    return messages


def call_openai_api(messages, model_name):
    for _ in range(5):
        try:
            api_result = client.chat.completions.create(
                model=model_name,
                messages=messages,
                logprobs=True,
                top_logprobs=20,
                max_tokens=1,
                temperature=0.0
            )
            break
        except Exception as e:
            print(e)
            print('TIMEOUT. Sleeping and trying again.')
            time.sleep(3)
    res = {}
    for top_logprob in api_result.choices[0].logprobs.content[0].top_logprobs:
        res[top_logprob.token] = top_logprob.logprob
    return res


if __name__ == '__main__':
    input_dataset_name = 'jmhessel/newyorker_caption_contest'
    input_task_name = 'matching'
    input_split = 'train'
    
    model_name = 'gpt-3.5-turbo'
    output_file_name = f'../nycartoon_data/nycartoon_{input_task_name}_split_{input_split}_text_label_logits.txt'

    dataset = list(load_dataset(input_dataset_name, input_task_name, split=input_split))
    messages = preprocess(dataset)
    
    if not os.path.exists(output_file_name):
        with open(output_file_name, 'w') as f:
            f.write('')
    
    with open(output_file_name, 'r') as f:
        existing_lines = f.readlines()
    existing_ids = set([line.split(" ", 1)[0] for line in existing_lines])
    
    for id, instance in tqdm(messages.items()):
        if id in existing_ids:
            print(f"Skipping {id} as it already exists in the output file.")
            continue
        prediction = dict()
        for caption, message in instance.items():
            logits = call_openai_api(
                message,
                model_name
            )
            prediction[caption] = logits

        with open(output_file_name, 'a') as f:
            f.write(f"{id} {json.dumps(prediction)}\n")