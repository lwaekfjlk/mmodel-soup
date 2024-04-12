import os
import ast
import pdb
import json

from tqdm import tqdm
import argparse
from openai import OpenAI

MODEL = "gpt-4-turbo-preview"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def query_openai(query):
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
    return json.dumps(res)
        

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--text_data", type=str, default=None, help='text_list')
    parser.add_argument("--save_file", type=str, default=None, help='save file path')
    parser.add_argument("--query", type=str, default=None, help='query')
    
    args = parser.parse_args()
    
    with open(args.text_data, "r") as f:
        input_str = f.read()
        lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
        text_lists = [ast.literal_eval(line) for line in lines]
    
    if not os.path.exists(args.save_file):
        with open(args.save_file, 'w') as f:
            f.write("")
    with open(args.save_file, 'r') as f:
        done_lines = f.readlines()
        done_image_names = [line.split(' ')[0] for line in done_lines]
    
    for text_list in tqdm(text_lists):
        image_name, sentence = text_list[0], text_list[1]
        if image_name in done_image_names:
            print(f"Skipping {image_name} because it is already done.")
            continue
        logits = query_openai(args.query + sentence)
        with open(args.save_file, 'a') as f:
            f.write(f"{image_name} {logits}\n")
            
if __name__ == "__main__":
    main()