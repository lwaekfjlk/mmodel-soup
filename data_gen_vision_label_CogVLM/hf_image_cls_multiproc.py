import os
import multiprocessing
from multiprocessing import Process, Lock
import time

import argparse
import torch
from tqdm import tqdm

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def process(num, lock, args, image_names):
    device = f"cuda:{7}"
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
            trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.from_pretrained,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True
        ).to(device).eval()
    
    begin_time = time.time()
    
    for i in range(len(image_names)):
        image_path = os.path.join(args.image_dir, image_names[i])
        image = Image.open(image_path).convert('RGB')
        query = "USER: {} ASSISTANT:".format(args.query)
        
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
        }
        
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]
        
        yes_token = tokenizer.tokenize("Yes")
        yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
        no_token = tokenizer.tokenize("No")
        no_token_id = tokenizer.convert_tokens_to_ids(no_token)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if args.quant:
                logits = logits.float()
            if args.fp16:
                logits = logits.half()
                
        last_logits = logits[:, -1, :]
        yes_logits = last_logits[:, yes_token_id]
        no_logits = last_logits[:, no_token_id]
        response = [yes_logits[0][0].item(), no_logits[0][0].item()]

        # with lock:
        with open(args.save_file, 'a') as f:
            f.write(f"{image_names[i].strip()} \"{response}\"\n")
        curr_time = time.time()
        time_to_finish = (curr_time - begin_time) / (i + 1) * (len(image_names) - i)
        class_result = "Yes" if yes_logits > no_logits else "No"
        print(f"proc {num} {i}/{len(image_names)} estimate time to finish {time_to_finish / 60:.2f} mins. Result: {response} Class: {class_result}")

def main():
    
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--image_dir", type=str, default=None, help='image dir')
    parser.add_argument("--save_file", type=str, default=None, help='save file path')
    parser.add_argument("--query", type=str, default=None, help='query')
    parser.add_argument("--num_processes", type=int, default=8, help='number of processes')

    args = parser.parse_args()
    
    image_names = os.listdir(args.image_dir)
    num_partitions = args.num_processes
    partition_size = len(image_names) // num_partitions
    print(f"total images: {len(image_names)} partition size: {partition_size}")
    
    with open(args.save_file, 'w') as f:
        f.write("")
    
    lock = Lock()
    processes = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == num_partitions - 1:
            end = len(image_names)
        partition = image_names[start:end]
        # p = Process(target=process, args=(i, lock, args, partition))
        # p.start()
        # processes.append(p)
        process(i, lock, args, partition)
    # for p in processes:
    #     p.join()
        
if __name__ == "__main__":
    main()