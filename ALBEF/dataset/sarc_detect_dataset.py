import json
import os
import ast
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class sarc_detect_train_dataset(Dataset):
    def __init__(self, text_file, transform, image_root, max_words=30):        
        with open(text_file, 'r') as f:
            input_str = f.read()
            lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
            self.text = [ast.literal_eval(line) for line in lines]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.text)
    

    def __getitem__(self, index):   
        image_id, sentence, label = self.text[index]
        image_path = os.path.join(self.image_root,'%s.jpg'%image_id)      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(sentence, self.max_words)

        return image, sentence, label


class sarc_detect_test_dataset(Dataset):
    def __init__(self, text_file, transform, image_root, max_words=30):        
        with open(text_file, 'r') as f:
            input_str = f.read()
            lines = [line.strip() for line in input_str.strip().split('\n') if line.strip()]
            self.text = [ast.literal_eval(line) for line in lines]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.text)
    

    def __getitem__(self, index):   
        image_id, sentence, label, ann_label = self.text[index]
        image_path = os.path.join(self.image_root,'%s.jpg'%image_id)      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(sentence, self.max_words)

        return image, sentence, ann_label
    