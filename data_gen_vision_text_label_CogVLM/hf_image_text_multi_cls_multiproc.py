import os
import json
import ast
import multiprocessing
from multiprocessing import Process, Lock
import time

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def process(num, lock, args, image_text_pairs):
    device = f"cuda:{num}"
    tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)
    
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    if args.quant:
        model = AutoModelForCausalLM.from_pretrained(
            args.from_pretrained,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
            cache_dir=args.cache_dir
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.from_pretrained,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True,
            cache_dir=args.cache_dir
        ).to(device).eval()
    
    begin_time = time.time()
    
    for i in range(len(image_text_pairs)):
        image_id, text = image_text_pairs[i]
        image_path = os.path.join(args.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        query = "USER: {} ASSISTANT:".format(args.query + " " + text if args.query is not None else text)
        
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
        }
        
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]
        
        # import pdb; pdb.set_trace()
        option_token_ids = {}
        for option in args.options:
            option_token = tokenizer.tokenize(option)
            option_token_id = tokenizer.convert_tokens_to_ids(option_token)
            assert len(option_token_id) == 1
            option_token_ids[option] = option_token_id[0]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if args.quant:
                logits = logits.float()
            if args.fp16:
                logits = logits.half()
                
        last_logits = logits[:, -1, :]

        logits_dict = {}
        class_result, class_result_logits = "", float('-inf')
        for option in args.options:
            option_token_id = option_token_ids[option]
            option_logits = last_logits[:, option_token_id]
            logits_dict[option] = option_logits.item()

            if class_result_logits < option_logits:
                class_result = option
                class_result_logits = option_logits
        
        result = {"image_id": image_id, "text": text, "response": logits_dict}
        result = json.dumps(result)
        
        with lock:
            with open(args.save_file, 'a') as f:
                f.write(f"{result}\n")
                
            curr_time = time.time()
            time_to_finish = (curr_time - begin_time) / (i + 1) * (len(image_text_pairs) - i)
            print(f"proc {num} {i}/{len(image_text_pairs)} estimate time to finish {time_to_finish / 60:.2f} mins. Result: {result} Class: {class_result}")

def main():
    
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--image_dir", type=str, default=None, help='image dir')
    parser.add_argument("--text_file", type=str, default=None, help='text file')
    parser.add_argument("--save_file", type=str, default=None, help='save file path')
    parser.add_argument("--query", type=str, default=None, help='query')
    parser.add_argument("--num_processes", type=int, default=8, help='number of processes')
    parser.add_argument("--options", type=str, default=None, help='options')
    parser.add_argument("--cache_dir", type=str, default=None, help='cache dir')

    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    args.options = ast.literal_eval(args.options)
    
    with open(args.text_file, 'r') as f:
        lines = f.readlines()
    
    image_text_pairs = []
    for line in lines:
        image_id, text = line.split(" ", 1)
        image_text_pairs.append((f"{image_id}.jpeg", text))
        
    num_partitions = args.num_processes
    partition_size = len(image_text_pairs) // num_partitions
    print(f"total images: {len(image_text_pairs)} partition size: {partition_size}")
    
    with open(args.save_file, 'w') as f:
        f.write("")
    
    lock = Lock()
    processes = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == num_partitions - 1:
            end = len(image_text_pairs)
        partition = image_text_pairs[start:end]
        p = Process(target=process, args=(i, lock, args, partition))
        p.start()
        processes.append(p)
        # process(i, lock, args, partition)
    for p in processes:
        p.join()
        
if __name__ == "__main__":
    main()