import os
import json
import multiprocessing
from multiprocessing import Process, Lock
import time

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def process(num, lock, args, quadriplets):
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
            device_map=device,
            cache_dir=args.cache_dir
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.from_pretrained,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True,
            device_map=device,
            cache_dir=args.cache_dir
        ).eval()
    begin_time = time.time()
    
    for i in range(len(quadriplets)):
        figurative_type, image_id, text, label = quadriplets[i]
        image_path = os.path.join(args.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        query = "USER: {} ASSISTANT:".format(args.query.format(text))
        
        if image is None:
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], template_version='base')
        else:
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
        }
        
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]
        
        token_A = tokenizer.tokenize(args.token_A)
        token_A_id = tokenizer.convert_tokens_to_ids(token_A)
        token_B = tokenizer.tokenize(args.token_B)
        token_B_id = tokenizer.convert_tokens_to_ids(token_B)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if args.quant:
                logits = logits.float()
            if args.fp16:
                logits = logits.half()
                
        last_logits = logits[:, -1, :]
        token_A_logits = last_logits[:, token_A_id]
        token_B_logits = last_logits[:, token_B_id]
        
        result = {"image_id": image_id, "text": text, "response": {args.token_A: token_A_logits.item(), args.token_B: token_B_logits.item()}, "label": label, "figurative_type": figurative_type}
        result = json.dumps(result)
        
        with lock:
            with open(args.save_file, 'a') as f:
                f.write(f"{result}\n")
                
            curr_time = time.time()
            time_to_finish = (curr_time - begin_time) / (i + 1) * (len(quadriplets) - i)
            class_result = args.token_A if token_A_logits > token_B_logits else args.token_B
            print(f"proc {num} {i}/{len(quadriplets)} estimate time to finish {time_to_finish / 60:.2f} mins. Result: {result} Class: {class_result}")

def main():
    
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--image_dir", type=str, default=None, help='image dir')
    parser.add_argument("--csv_file", type=str, default=None, help='text file')
    parser.add_argument("--save_file", type=str, default=None, help='save file path')
    parser.add_argument("--query", type=str, default=None, help='query')
    parser.add_argument("--num_processes", type=int, default=8, help='number of processes')
    parser.add_argument("--token_A", type=str, default=None, help='token A')
    parser.add_argument("--token_B", type=str, default=None, help='token B')
    parser.add_argument("--cache_dir", type=str, default=None, help='cache dir')

    args = parser.parse_args()
    
    with open(args.csv_file, 'r') as f:
        lines = f.readlines()
    
    quadriplets = []
    for i in range(1, len(lines)):
        line = lines[i]
        try:
            figurative_type, image_id, text, label = line.split(",")
        except:
            import pdb; pdb.set_trace()
        quadriplets.append((figurative_type.strip(), f"{image_id}.jpeg", text, label.strip()))
        
    num_partitions = args.num_processes
    partition_size = len(quadriplets) // num_partitions
    print(f"total images: {len(quadriplets)} partition size: {partition_size}")
    
    with open(args.save_file, 'w') as f:
        f.write("")
    
    lock = Lock()
    processes = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == num_partitions - 1:
            end = len(quadriplets)
        partition = quadriplets[start:end]
        p = Process(target=process, args=(i, lock, args, partition))
        p.start()
        processes.append(p)
        # process(i, lock, args, partition)
    for p in processes:
        p.join()
        
if __name__ == "__main__":
    main()