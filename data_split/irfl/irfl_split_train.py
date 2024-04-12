import jsonlines
from collections import defaultdict

def get_prediction():
    prediction = defaultdict(dict)
    with jsonlines.open('../../irfl_data/intermediate_data/irfl_vision_text_label_logits.jsonl', 'r') as f:
        