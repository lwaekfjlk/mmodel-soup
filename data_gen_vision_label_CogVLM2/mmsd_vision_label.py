import os
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (collate_fn, load_ground_truth_labels, load_images,
                   prepare_input_samples, recur_move_to, save_results)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "./cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
device = "cuda"

image_folder = "../sarc_data/data_raw/subfolder_4"
data_folder = "../sarc_data/data_raw"
output_file = (
    "../sarc_data/data_gen_output/sarc_image_only_pred_cogvlm2_subfolder_4.jsonl"
)

print(image_folder)
print(output_file)

batch_size = 4
query = (
    "Please analyze the image provided for sarcastic or not. The image is the screenshot of the image in a twitter. It might include a lot of text so you need to combine the information of the text in the image."
    "If you think the image includes exaggerated emotions (like laughing or looking angry or raising bows) or exaggerated posture (like stretching hands), please answer 'Yes'."
    "If you think the image includes text that is sarcastic or exaggerated, please answer 'Yes'."
    "If you think the image shows people discussing serious things and just daily routines, please answer 'No'."
    "You need to think about what is the potential even going on in the image."
    "Please make sure that your answer is based on the image itself, not on the context or your personal knowledge."
    "There are only two options: 'Yes' or 'No'."
    "If you are not sure, please provide your best guess and do not say that you are not sure."
    "You should only make No judgement when you are very sure that the text is not sarcastic. As long as you think potentially it is sarcastic, you should say Yes."
)

ground_truth_labels = load_ground_truth_labels(
    data_folder,
    ["sarc_dataset_train.json", "sarc_dataset_val.json", "sarc_dataset_test.json"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

yes_token = tokenizer.tokenize("Yes")
yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
no_token = tokenizer.tokenize("No")
no_token_id = tokenizer.convert_tokens_to_ids(no_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True, device_map=device
).eval()

data = load_images(image_folder)

length = len(data)
print(length)
print(len(ground_truth_labels))

for idx in tqdm(
    range(0, length, batch_size), desc="Processing", total=length // batch_size
):
    i_list = [data[idx + i]["image"] for i in range(batch_size) if idx + i < length]

    input_sample_list, image_id_list = prepare_input_samples(
        model, tokenizer, query, i_list
    )

    start = time.time()
    input_batch = collate_fn(input_sample_list, tokenizer)
    input_batch = recur_move_to(
        input_batch, device, lambda x: isinstance(x, torch.Tensor)
    )
    input_batch = recur_move_to(
        input_batch,
        torch.bfloat16,
        lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
    )
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
