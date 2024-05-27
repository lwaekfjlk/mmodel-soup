import os
import multiprocessing
from multiprocessing import Process, Lock
import time
import json

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

def process(num, lock, args, image_names):
    num_gpus = torch.cuda.device_count()
    device = f"cuda:{num % num_gpus}"
    
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

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                        "do_sample": False} # "temperature": 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
        
        with lock:
            result = json.dumps({"image_id": image_names[i], "description": response.strip()})
            with open(args.save_file, 'a') as f:
                f.write(f"{result}\n")
            
            curr_time = time.time()
            time_to_finish = (curr_time - begin_time) / (i + 1) * (len(image_names) - i)
            print(f"proc {num} {i}/{len(image_names)} estimate time to finish {time_to_finish / 60:.2f} mins. Result: {result}")

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
    
    existing_ids = []
    if not os.path.exists(args.save_file):
        with open(args.save_file, 'w') as f:
            f.write("")
    else:
        with open(args.save_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            existing_ids.append(data['image_id'])
    
    image_names = os.listdir(args.image_dir)
    # import pdb; pdb.set_trace()
    image_names = [x for x in image_names if x not in existing_ids]
    if ".gitkeep" in image_names:
        image_names.remove(".gitkeep")
    print(f"{len(existing_ids)} ids already processed, {len(image_names)} ids to process")
    num_partitions = args.num_processes
    partition_size = len(image_names) // num_partitions
    print(f"total images: {len(image_names)} partition size: {partition_size}")
    
    lock = Lock()
    processes = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == num_partitions - 1:
            end = len(image_names)
        partition = image_names[start:end]
        p = Process(target=process, args=(i, lock, args, partition))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
if __name__ == "__main__":
    main()