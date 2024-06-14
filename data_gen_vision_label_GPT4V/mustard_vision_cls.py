import os
import ast
import pdb
import json

from tqdm import tqdm
import time
import argparse
import base64
from openai import OpenAI

MODEL = "gpt-4-vision-preview"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_openai(image_path, query):
    base64_image = encode_image(image_path)
    attempts = 0
    success = False
    while attempts < 5 and not success:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                temperature=0.0,
            )
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            tokens = [top_logprob.token for top_logprob in top_logprobs]
            if 'Yes' in tokens and 'No' in tokens:
                success = True
            else:
                success = False
            attempts += 1
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts < 5:
                time.sleep(5)
            else:
                print("Failed after 5 attempts.")

        if success:
            break

    res = {}
    for top_logprob in response.choices[0].logprobs.content[0].top_logprobs:
        res[top_logprob.token] = top_logprob.logprob
    return json.dumps(res)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='../mustard_data/data_raw/images', help='image folder position')
    parser.add_argument("--text_data", type=str, default='../mustard_data/data_raw/mustard_raw_data_speaker_independent_test.json', help='text_list')
    parser.add_argument("--save_file", type=str, default='../mustard_data/data_gen_output/mustard_vision_only_pred_test.json', help='save file path')
    
    args = parser.parse_args()

    query = 'Are the people in the image being sarcastic or not? You need to think based on their figurative language, body language, face emotion. Sarcasm often happens when people have intensive feelings or emotions. Answer with "Yes" or "No". Follow your initial judgement and explain why.'

    with open(args.text_data, 'r') as f:
        dataset = json.load(f)
    
    with open(args.save_file, 'w') as f:
        f.write("")
    
    for id in tqdm(dataset.keys()):
        img_name = id + '.jpg'
        img_path = os.path.join(args.image_path, img_name)
        logits = query_openai(img_path, query)
        with open(args.save_file, 'a') as f:
            f.write(f"{id} {logits}\n")

            
if __name__ == "__main__":
    main()