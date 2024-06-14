import os
import ast
import pdb
import json

from tqdm import tqdm
import time
import argparse
import base64
from openai import OpenAI

MODEL = "gpt-4-turbo-preview"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def query_openai(query):
    attempts = 0
    success = False
    while attempts < 5 and not success:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": query}
            ],
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
            temperature=0.0
        )
        res = {}
        for top_logprob in response.choices[0].logprobs.content[0].top_logprobs:
            res[top_logprob.token] = top_logprob.logprob
        if 'Yes' in res and 'No' in res.keys():
            success = True
        else:
            success = False
        attempts += 1
    return json.dumps(res)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_data", type=str, default='../mustard_data/data_raw/mustard_raw_data_speaker_independent_test.json', help='text_list')
    parser.add_argument("--save_file", type=str, default='../mustard_data/data_gen_output/mustard_text_only_pred_test.json', help='save file path')
    
    args = parser.parse_args()

    query = 'Are the people in the image being sarcastic or not? You need to think based on their figurative language, body language, face emotion. Sarcasm often happens when people have intensive feelings or emotions. Answer with "Yes" or "No". Follow your initial judgement and explain why.'

    with open(args.text_data, 'r') as f:
        dataset = json.load(f)
    
    with open(args.save_file, 'w') as f:
        f.write("")
    
    for id in tqdm(dataset.keys()):
        utterance = dataset[id]['utterance']
        speaker = dataset[id]['speaker']
        context = dataset[id]['context']
        context_speaker = dataset[id]['context_speakers']
        context_conversation = ''
        for con, con_speaker in zip(context, context_speaker):
            context_conversation += f"{con_speaker} says that {con} "
        query = f"The context for the conversation are {context_conversation} The final uttearnce is that {speaker} says that {utterance} Please based on the context information and check whether the utterance is sarcastic or not. Answer with 'Yes' or 'No'. Follow your initial judgement and explain why."
        logits = query_openai(query)
        with open(args.save_file, 'a') as f:
            f.write(f"{id} {logits}\n")

            
if __name__ == "__main__":
    main()