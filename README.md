# mmodel-soup

The file structure should be like:

1. unimodal_label --> y1 and y2 unimodal label collection
2. image_caption_generation --> project vision information into text
3. dataset_splitting --> split dataset into multiple parts
4. expert_model_training (sub-folers: LLM + MLLM + MM) --> train the model based on multiple components
5. expert_model_inference (sub=folers: LLM + MLLM + MM) --> check the inference part of the dataset
