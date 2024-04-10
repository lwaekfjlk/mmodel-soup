import csv
import os
import ast
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption

label_map = {"Figurative": 1, "Not Figurative": 0}

class irfl_train_dataset(Dataset):
    def __init__(self, csv_file, transform, image_root, max_words=30):        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.text = [row for row in reader][1:] # the first row is the header
        # import pdb; pdb.set_trace()
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.text)
    

    def __getitem__(self, index):   
        image_id, sentence, label = self.text[index]
        image_path = os.path.join(self.image_root,'%s.jpeg'%image_id)      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(sentence, self.max_words)

        return image, sentence, label_map[label]


class irfl_test_dataset(Dataset):
    def __init__(self, csv_file, transform, image_root, max_words=30):        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.text = [row for row in reader][1:] # the first row is the header
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):   
        image_id, sentence, label = self.text[index]
        image_path = os.path.join(self.image_root,'%s.jpeg'%image_id)      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(sentence, self.max_words)

        return image, sentence, label_map[label], image_id
    