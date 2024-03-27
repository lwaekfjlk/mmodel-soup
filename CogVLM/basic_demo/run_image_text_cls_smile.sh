python hf_image_text_cls_multiproc.py --from_pretrained THUDM/cogagent-chat-hf --bf16 --image_dir ../../irfl_data/image_data --text_file ../../irfl_data/simile_data/image_text_pairs.txt --save_file ../../irfl_data/simile_data/image_text_cls.txt --query "Categorize the relationship between the image and the provided text by choosing the most fitting option:\nA. Partial Literal: Some objects/ entities of the phrase are visualized.\nB. Figurative: The image conveys one or more definitions of the idiom.\nDetermine whether the image's content is a direct visual match for the text's elements or if it represents the text's ideas through symbolic meanings. Here is the provided text: " --token_A "A" --token_B "B" --num_processes 1