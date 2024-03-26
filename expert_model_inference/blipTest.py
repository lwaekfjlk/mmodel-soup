import ipdb
import ast
import os
import csv
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, Blip2ForConditionalGeneration, BlipImageProcessor
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

if __name__ == '__main__':
    # path to data  
    test_path = '/root/bak/mmodel-soup/dataset/sarcasm_blip_test.csv'
    dataset = load_dataset("csv", data_files=test_path)['train']
#   model_path = "/root/bak/CogVLM/blip_output/model"
#   model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model.to(device)
    ipdb.set_trace()
    for example in dataset:
        text = example['text']
        image = example['image']
        label = example['label']
            


