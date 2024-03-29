import os
import random
from typing import List, Dict, Union
import logging
from tqdm import tqdm
import asyncio
from openai_prompting import generate_from_openai_chat_completion
import argparse
from mustard_utils import *


PROMPT = """
You are a sarcasm detection agent. You are given a conversation between multiple characters in the TV series including The Big Bang Theory and Friends. Based on the context information in the conversation, you need to determine whether the final utterance said by a character is sarcasm or not. 
You should answer with "Reason: xxx, Answer: x", where xxx is the reason, and x is a number from -5 to 5 but not 0, -5 means that it is obvious not sarcasm, 5 means it is obvious sarcasm, 0 means that it is not either sarcasm or not sarcasm, 1 means it is not quite sure sarcasm, and -1 means it is not quite sure not sarcasm. You should not predict 0.
Let me explain more about the sarcasm prediction task.
Sarcasm is when people don't respond directly but respond in an unexpected way with a special intention outside of the sentence itself and this special intention is to show their unwillingness or dissatisfaction. Not sarcasm is when the response doesn't include any intentional information and just wants to say what the response means. And the response does not carry any unwillingness or dissatisfaction.
Let me explain more about the information that you are using.
You are given a person's utterance said by a character in the TV series. Moreover, you are given the context of the conversation which provides information related to why the character says that utterance. Additionally, you are given a tone description of the tone sound when the character says the utterance.  
"""

DISAGREE_PROMPT = 'The tone and the conversation context gives contradictory information related to sarcasm judgment. Therefore, you need to think step-by-step and think very carefully how those information combine together and indicate.'
AGREE_PROMPT = 'Information provided might include redundant information for sarcasm judgment. Think why the character said that.'
NORMAL_PROMPT = 'Think why the character said that.'

def generate_context(context_speaker: List[str], context: List[str]) -> List[str]:
    """Generates context lines from given speakers and their utterances."""
    return [f'{s}: "{u}"' for s, u in zip(context_speaker, context)]

def audio_emotion(utterance_speaker: str, utterance: str, emotion: str) -> str:
    """Generates information about audio modality."""
    audio_emotion_prompts = {
        'neutral': 'When it comes to audio, no strong emotion is detected.',
        'happy': 'When it comes to audio, the utterance is said with a happy tone.',
        'angry': 'When it comes to audio, the utterance is said with an angry tone.',
        'sad': 'When it comes to audio, the utterance is said with a sad tone.'
    }
    return f'When it comes to the utterance {utterance_speaker} \"{utterance}\" {audio_emotion_prompts[emotion]}'

def face_emotion(vision: List[str]) -> str:
    """Generates information about vision modality."""
    if len(vision) == 0:
        return 'During saying this utterance, no strong facial emotion like happy, angry, sad, surprise is detected.'

    vision_prompt = 'During saying this utterance, '
    for vid, info in enumerate(vision):
        split_emotions = info.split(';')[:-1]
        for eid, emotion in enumerate(split_emotions):
            if eid == 0:
                vision_prompt += f'someone in the scene becomes {emotion}. '
            else:
                vision_prompt += f'Meanwhile, another person in the scene becomes {emotion}. '

        if vid != len(vision) - 1:
            vision_prompt += 'After that, '
    return vision_prompt

def disagreement_information(agree_or_not: str) -> str:
    """Generates disagreement/agreement information."""
    if agree_or_not == 'disagree':
        return DISAGREE_PROMPT
    elif agree_or_not == 'agree':
        return AGREE_PROMPT
    elif agree_or_not == 'full':
        return NORMAL_PROMPT
    else:
        raise ValueError(f'Invalid agree_or_not value: {agree_or_not}')

def build_example(args: object, data: Dict[str, Union[str, List[str]]], emotion: str, vision: str, is_first: bool) -> str:
    """Construct an example question based on the provided data and emotion."""
    utterance = data['utterance']
    context = data['context']
    utterance_speaker = data['speaker']
    context_speaker = data['context_speakers']

    question_lines = ["Here is one example of how you can learn how to judge sarcasm."] if is_first else ["OK, thanks for your answer. Here is another example of judging sarcasm."]

    if 'text' in [args.modality1, args.modality2]:
        question_lines.extend(generate_context(context_speaker, context))


    if 'text' in [args.modality1, args.modality2]:
        question_lines.append(f'{utterance_speaker}: "{utterance}"')

    if 'audio' in [args.modality1, args.modality2]:
        if args.use_video_llama:
            question_lines.append(emotion)
        else:
            question_lines.append(audio_emotion(utterance_speaker, utterance, emotion))

    if 'vision' in [args.modality1, args.modality2]:
        if args.use_video_llama:
            question_lines.append(vision)
        else:
            question_lines.append(face_emotion(vision))

    if args.multimodal:
        question_lines.append(disagreement_information(args.agree_or_not))

    question_lines.append(f'Please judge the utterance spoken by {utterance_speaker}: "{utterance}" is sarcasm or not.')
    return "\n".join(question_lines)


def build_answer(args: object, answer: str, data: str, audio: str, vision: str) -> str:
    """
    Construct an answer based on the provided answer.
    
    Args:
        answer: The answer provided by the expert.
        
    Returns:
        The constructed answer as a string.
    """
    utterance = data['utterance']
    context = data['context']
    utterance_speaker = data['speaker']
    context_speaker = data['context_speakers']

    question_lines = ["Given this conversation:"]

    if 'text' in [args.modality1, args.modality2]:
        question_lines.extend(generate_context(context_speaker, context))

    if 'text' in [args.modality1, args.modality2]:
        question_lines.append(f'{utterance_speaker}: "{utterance}"')

    if 'audio' in [args.modality1, args.modality2]:
        if args.use_video_llama:
            question_lines.append(audio)
        else:
            question_lines.append(audio_emotion(utterance_speaker, utterance, audio))

    if answer is True:
        question_lines.append(f'Describe why this utterance "{utterance}" is sarcasm in one sentence. You need to first generate one sentence with a strict format of "Reason: xxx" based on tone emotion and conversation information, secondly, you need to generate a confidence score from 1 to 5 to indicate that whether you are sure that this is sarcasm. 5 indicates that you strongly feel that it is sarcasm and you can tell clear reason why it is that and 1 indicates that you almost dont feel any certainty and you cannot speak any clear reason. Answer with the format of "Answer: xxx", xxx is the confidence number. You should not choose 5 unless you are very confident. You need to mention audio emotion and conversation information in your reason.')
    else:
        question_lines.append(f'Describe why this utterance "{utterance}" is not sarcasm in one sentence. You need to first generate one sentence with a strict format of "Reason: xxx" based on tone emotion and conversation information, secondly, you need to generate a confidence score from -1 to -5 to indicate that whether you are sure that this is sarcasm. -5 indicates that you strongly feel that it is not sarcasm and you can tell clear reason why it is that and -1 indicates that you almost dont feel any certainty and you cannot speak any clear reason. Answer with the format of "Answer: xxx", xxx is the confidence number. You should not choose -5 unless you are very confident. You need to mention audio emotion and conversation information in your reason.')

    if answer is True:
        prompts = [[{'role': 'user', 'content': '\n'.join(question_lines)}]]
    else:
        prompts = [[{'role': 'user', 'content': '\n'.join(question_lines)}]]

    reasons = asyncio.run(generate_from_openai_chat_completion(
        contexts=prompts,
        model='gpt-4',
        temperature=1.0,
        max_tokens=1024,
        top_p=0.8,
        n=1,
        requests_per_minute=args.model_requests_per_minute,
    ))
    return reasons



def select_few_shot_examples(args: object, dataset: Dict[int, Dict[str, Union[str, List[str]]]], speaker: str) -> List[int]:
    """
    Selects few-shot examples from the dataset.
    
    Args:
        args: A dictionary containing various arguments such as number of few-shot examples and testing mode.
        dataset: A dictionary of data with conversation details.
        
    Returns:
        A list of indices of few-shot examples.
    """
    available_pos_candidates = {}
    available_neg_candidates = {}
    for id, data in dataset.items():
        if data['sarcasm'] == True and data['speaker'] == speaker:
            available_pos_candidates[id] = data
        if data['sarcasm'] == False and data['speaker'] == speaker:
            available_neg_candidates[id] = data

    random.seed(43)
    try:
        pos_few_shot = random.sample(list(available_pos_candidates.keys()), args.few_shot_example_num // 2)
    except:
        pos_few_shot = random.sample(list(dataset.keys()), args.few_shot_example_num // 2)
    try:
        neg_few_shot = random.sample(list(available_neg_candidates.keys()), args.few_shot_example_num - args.few_shot_example_num // 2)
    except:
        neg_few_shot = random.sample(list(dataset.keys()), args.few_shot_example_num - args.few_shot_example_num // 2)
    return pos_few_shot + neg_few_shot


def generate_few_shot_answer(args: object, answer: str) -> str:
    answer_dict = {True: 'yes', False: 'no'}
    if args.answer_with_confidence:
        return f'Answer: {answer_dict[answer[0]]} Confidence: 5'
    elif args.answer_with_rationale:
        return f'Answer: {answer_dict[answer[0]]} Rationale: {answer[1]}'
    else:
        return f'Answer: {answer}'


def construct_few_shot_examples(args, dataset, labels, test_data, audio_emotion, face_emotion) -> List[Dict[str, str]]:
    """
    Construct few-shot examples from the dataset.
    
    Args:
        args: A dictionary containing various arguments such as number of few-shot examples and testing mode.
        dataset: A dictionary of data with conversation details.
        labels: A dictionary mapping data indices to their labels (sarcasm or not).
        
    Returns:
        A list of few-shot examples.
    """
    few_shot_examples = [{'role': 'system', 'content': PROMPT}]
    speaker = test_data['speaker']
    answers_dict = {}
    # read answers_dict
    if os.path.exists('audio_text_few_shot_answers.json'):
        with open('audio_text_few_shot_answers.json', 'r') as f:
            answers_dict = json.load(f)


    few_shot_keys = []
    for id in answers_dict.keys():
        if id in dataset.keys() and dataset[id]['speaker'] == speaker:

            few_shot_keys.append(id)
    
    if len(few_shot_keys) < args.few_shot_example_num:
        # random sample

        few_shot_keys = select_few_shot_examples(args, dataset, speaker)
    else:
        few_shot_keys = random.sample(few_shot_keys, args.few_shot_example_num)

    assert len(few_shot_keys) == args.few_shot_example_num
    

    for id, few_shot_idx in enumerate(few_shot_keys):
        few_shot_data = dataset[few_shot_idx]
        few_shot_audio_emotion = audio_emotion[few_shot_idx]
        few_shot_face_emotion = face_emotion[few_shot_idx]
        few_shot_answer = labels[few_shot_idx]
        question = build_example(args, few_shot_data, few_shot_audio_emotion, few_shot_face_emotion, is_first=(id == 0))
        if few_shot_idx in answers_dict.keys():
            answer = answers_dict[few_shot_idx]
        else:
            answer = build_answer(args, few_shot_answer, few_shot_data, few_shot_audio_emotion, few_shot_face_emotion)[0]
            answers_dict[few_shot_idx] = answer
        few_shot_examples.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ])


    with open('audio_text_few_shot_answers.json', 'w') as f:
        json.dump(answers_dict, f)


    return few_shot_examples


def build_prompt(args: Dict[str, Union[int, bool]], train_dataset: Dict[int, Dict[str, Union[str, List[str]]]], test_dataset: Dict[int, Dict[str, Union[str, List[str]]]], train_labels: Dict[int, str], test_labels: Dict[int, str]) -> Dict[int, List[Dict[str, str]]]:
    """
    Generate prompts based on the dataset and few-shot examples.
    
    Args:
        args: A dictionary containing various arguments such as number of few-shot examples and testing mode.
        dataset: A dictionary of data with conversation details.
        labels: A dictionary mapping data indices to their labels (sarcasm or not).
        audio_prompts: A dictionary mapping emotions to specific prompts.
        
    Returns:
        A dictionary of prompts.
    """

    if args.use_video_llama:
        audio_emotion = load_audio_description(args)
        face_emotion = load_vision_description(args)
    else:
        audio_emotion = load_audio_emotion(args)
        face_emotion = load_face_emotion(args)

    prompts = {}
    all_keys = list(test_dataset.keys())


    # Construct the main prompts using the pre-built few-shot examples
    for idx in tqdm(all_keys):
        test_data = test_dataset[idx]
        few_shot_messages = construct_few_shot_examples(args, train_dataset, train_labels, test_data, audio_emotion, face_emotion) 
        messages = few_shot_messages.copy()
        emotion = audio_emotion[idx]
        vision = face_emotion[idx]
        question = build_example(args, test_data, emotion, vision, is_first=False)
        messages.append({'role': 'user', 'content': question})
        prompts[idx] = messages


        # Break early if in testing mode
        if args.testing_mode and len(prompts) >= args.testing_num:
            break

        if args.verbose:
            for msg in messages:
                print(msg['content'])
    return prompts


def expert_generate(args: object, prompts: dict) -> dict:
    """
    Generates expert answers from provided prompts using the OpenAI API.
    - Keeps track of prompts that need to be re-queried.
    - Updates results based on successful responses.
    """
    results = {id: {'prediction': None, 'confidence': None} for id in prompts.keys()}
    iter_num = 0

    while iter_num <= args.model_generate_max_iter_num and prompts:
        iter_num += 1
        answers = asyncio.run(generate_from_openai_chat_completion(
            contexts=list(prompts.values()),
            model=args.model_name,
            temperature=args.model_temperature,
            max_tokens=args.model_max_output_len,
            top_p=args.model_top_p,
            n=args.model_return_num,
            requests_per_minute=args.model_requests_per_minute,
        ))

        to_remove = []  # List to track keys that should be removed after loop iteration
        for id, ans in zip(prompts.keys(), answers):
            pred, pred_number, conf = filter_invalid_ans(args, ans)
            if args.predict_confidence:
                if pred is not None and conf is not None:
                    results[id]['prediction'] = pred
                    results[id]['prediction_range'] = pred_number
                    results[id]['confidence'] = conf
                    results[id]['raw_answer'] = ans
                    results[id]['prompt'] = prompts[id]
                    to_remove.append(id)  # Add to removal list instead of directly removing
            else:
                if pred is not None:
                    results[id]['prediction'] = pred
                    results[id]['prediction_range'] = pred_number
                    results[id]['raw_answer'] = ans
                    results[id]['prompt'] = prompts[id]
                    to_remove.append(id)

        # Now, remove the items
        for id in to_remove:
            prompts.pop(id)

    return results


def split_into_train_test_folds(data_dict, label_dict, num_folds=5):
    # Ensure data and label dictionaries have the same keys
    assert data_dict.keys() == label_dict.keys(), "Data and label dictionaries must have the same keys"
    
    # Convert the dictionary into a list of (key, value) tuples
    items = list(data_dict.items())
    
    # Shuffle the items for a random split
    random.shuffle(items)
    
    # Split the items into num_folds roughly equal parts
    fold_size = len(items) // num_folds
    data_folds = [dict(items[i * fold_size: (i + 1) * fold_size]) for i in range(num_folds)]
    
    # Handle any remaining items due to rounding errors in partitioning
    for i, item in enumerate(items[num_folds * fold_size:]):
        data_folds[i].update([item])
    
    # Generate label folds based on the keys of the data folds
    label_folds = [{key: label_dict[key] for key in fold} for fold in data_folds]

    # Create train-test splits for cross-validation
    train_test_splits = []
    for i in range(num_folds):
        test_data = data_folds[i]
        test_labels = label_folds[i]

        train_data = {}
        train_labels = {}

        for j in range(num_folds):
            if j != i:
                train_data.update(data_folds[j])
                train_labels.update(label_folds[j])

        train_test_splits.append(((train_data, train_labels), (test_data, test_labels)))

    return train_test_splits


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.getLogger().addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_mode', action='store_true')
    parser.add_argument('--testing_num', type=int, default=30)
    parser.add_argument('--dataset_directory', type=str, default='./')
    parser.add_argument('--dataset_filename', type=str, default='sarcasm_data.json')
    parser.add_argument('--speaker_independent', type=bool, default=False)
    parser.add_argument('--use_subpart', action='store_true')
    parser.add_argument('--subpart_directory', type=str, default='./')
    parser.add_argument('--agree_or_not', type=str, choices=['full', 'agree', 'disagree'], default='full')
    parser.add_argument('--vision_info_directory', type=str, default='./data/frames/')
    parser.add_argument('--vision_info_filename', type=str, default='face_emotions.csv')
    parser.add_argument('--audio_info_directory', type=str, default='./data/audios/')
    parser.add_argument('--audio_info_filename', type=str, default='utterances_final_filtered.csv')
    parser.add_argument('--output_directory', type=str, default='./audio_text_results/')
    parser.add_argument('--multimodal', action='store_true')
    parser.add_argument('--modality1', type=str, choices=['text', 'audio', 'vision'], default=None)
    parser.add_argument('--modality2', type=str, choices=['text', 'audio', 'vision'], default=None)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model_name', type=str, default='gpt-4')
    parser.add_argument('--model_max_output_len', type=int, default=512)
    parser.add_argument('--model_temperature', type=float, default=1.0)
    parser.add_argument('--model_top_p', type=float, default=1.0)
    parser.add_argument('--model_requests_per_minute', type=int, default=100)
    parser.add_argument('--model_generate_max_iter_num', type=int, default=50)
    parser.add_argument('--model_return_num', type=int, default=1)
    parser.add_argument('--few_shot_example_num', type=int, default=4)
    parser.add_argument('--predict_confidence', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--answer_with_rationale', action='store_true')
    parser.add_argument('--answer_with_confidence', action='store_true')
    parser.add_argument('--use_video_llama', action='store_true')
    parser.add_argument('--audio_description_directory_from_video_llama', type=str, default='./')
    parser.add_argument('--audio_description_filename_from_video_llama', type=str, default='sarcasm_audio.json')
    parser.add_argument('--vision_description_directory_from_video_llama', type=str, default='./')
    parser.add_argument('--vision_description_filename_from_video_llama', type=str, default='sarcasm_vision.json')
    args = parser.parse_args()

    train_dataset, test_dataset, train_labels, test_labels = load_dataset(args)

    train_test_splits = split_into_train_test_folds(train_dataset, train_labels, num_folds=5)
    if args.speaker_independent is False:
        for train_test_split in train_test_splits:
            train_dataset, train_labels = train_test_split[0]
            test_dataset, test_labels = train_test_split[1]
            prompts = build_prompt(args, train_dataset, test_dataset, train_labels, test_labels)
            results = expert_generate(args, prompts)
            write_output(args, results)
            metric_results = eval_metric(results, test_labels)
            print(metric_results)
    else:
        prompts = build_prompt(args, train_dataset, test_dataset, train_labels, test_labels)
        results = expert_generate(args, prompts)
        write_output(args, results)
        metric_results = eval_metric(results, test_labels)
        print(metric_results)