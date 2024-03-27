from torch.utils.data import Dataset
from dataset.utils import pre_caption


class bi_cls_dataset(Dataset):
    def __init__(self, dataset, transform, max_words=1000):
        self.dataset = dataset
        self.transform = transform
        self.max_words = max_words
        self.labels = {'A':1, 'B':0}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        data_point = self.dataset[index]
        image = self.transform(data_point['image'])
        text = "A: {} B: {}".format(data_point['caption_choices'][0], data_point['caption_choices'][1])
        sentence = pre_caption(text, self.max_words)
        
        return image, sentence, self.labels[data_point['label']]
    