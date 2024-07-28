import os
import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import recur_move_to, collate_fn, load_ground_truth_labels, load_images, prepare_input_samples, save_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/dataset/granite_ckpt/haofeiyu/cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
device = 'cuda'

image_folder = "../urfunny_data/data_raw/images"
data_folder = "../urfunny_data/data_raw"
output_file = "../urfunny_data/data_gen_output/urfunny_image_only_pred_cogvlm2.jsonl"

batch_size = 4
query = (
    "You are looking at a screenshot of a TED talk. It is part of the talk and it can be a slide or a speaker."
    "Please analyze the image provided the answer to show whether the image is part of a talk that is showing serious content or trying to show some potentially funny content that can make audience laugh."
    "If you are looking at a slide, please think about the content of the slide."
    "If the slide is showing some very interesting and informal things, we believe the speaker is trying to make some jokes and please answer 'Yes'."
    "If the slide is showing some very serious and formal things, we believe the speaker is trying to show some serious content and please answer 'No'."
    "If you are looking at a speaker, please think about the speaker's facial expression and body language."
    "If you think the image includes exaggerated emotions or its body language is exaggerated. We believe the speaker is talking about some informal things and please answer 'Yes'."
    "If you think the speaker in the image looks very serious and formal, they are tring to convey their key points and please answer 'No'."
    "Please make sure that your answer is based on the image itself, not on the context or your personal knowledge."
    "There are only two options: 'Yes' or 'No'."
    "If you are not sure, please provide your best guess and do not say that you are not sure."
)

ground_truth_labels = load_ground_truth_labels(data_folder, ["urfunny_dataset_train.json", "urfunny_dataset_val.json", "urfunny_dataset_test.json"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

yes_token = tokenizer.tokenize("Yes")
yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
no_token = tokenizer.tokenize("No")
no_token_id = tokenizer.convert_tokens_to_ids(no_token)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True, device_map=device).eval()

data = load_images(image_folder)

length = len(data)
print(length)
print(len(ground_truth_labels))

for idx in tqdm(range(0, length, batch_size), desc="Processing", total=length // batch_size):
    i_list = [data[idx + i]["image"] for i in range(batch_size) if idx + i < length]

    input_sample_list, image_id_list = prepare_input_samples(model, tokenizer, query, i_list)

    start = time.time()
    input_batch = collate_fn(input_sample_list, tokenizer)
    input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
    input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
    print(f"Prepare batch time: {time.time() - start}")

    start = time.time()
    with torch.no_grad():
        outputs = model(**input_batch)
        logits = outputs.logits.to("cpu").float()
        for i in range(len(logits)):
            last_logits = logits[i, -1, :]
            softmax_logits = torch.nn.functional.softmax(last_logits, dim=0)
            yes_logits = softmax_logits[yes_token_id]
            no_logits = softmax_logits[no_token_id]
            response = {"Yes": yes_logits[0].item(), "No": no_logits[0].item()}
            prediction = 1 if yes_logits[0].item() > no_logits[0].item() else 0
            if image_id_list[i] in ground_truth_labels:
                gth = ground_truth_labels[image_id_list[i]]
            else:
                gth = None
            save_results(output_file, image_id_list[i], response, gth, prediction)
    print(f"Inference time: {time.time() - start}")
