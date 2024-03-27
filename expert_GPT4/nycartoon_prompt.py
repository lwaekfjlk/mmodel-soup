def prompt_construct(
    scene,
    entities,
    image_description,
    image_uncanny_description,
    caption,
):
    prompt = f"""
You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. 
You follow the contest each week, and understand what makes for a humorous caption for each cartoon. 
You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.
You will be given a cartoon description and a caption choice. You need to decide whether the caption is humorous or not.
You are able to detect what is the meaning of the caption when you read the description for the cartoon.
You are able to recognize whether the caption can explain the uncanny point in the cartoon.
You are able to recognize whether the caption combining with cartoon is humorous or not.

You need to think step-by-step. 
(1) you need to think what is the words like 'you', 'he', 'it' or other nouns in the caption pointing to in the cartoon. 
(2) you need to connect with the information from the previous step, predict new information based on caption and image. 
(3) you need to think whether the new information can explain the uncanny point in the cartoon or not. 
(4) answer with the format of "x", x is either 'yes' or 'no'.
"""
    example1 = f"""
Here is one example that you can learn how to detect humor.
Cartoon content: 
The scene is clouds. 
Entities mentioned in the cartoon are God, Lightning_strike. 
Two men are standing on clouds, one is dressed in a suit and the other a robe. They are both holding lightning bolts, but the man in the suit looks to be about to throw his and the man in the robe is angrily looking on while holding his by his waist. 
The uncanny point is that Seeing two men standing on clouds and holding/throwing lightning bolts is unusual. Additionally, having one of the men dressed in a suit and the other a robe is also unexpected. The possible discontent felt by one of the men towards the other during the event/activity is also curious.

Caption content: 
Business school changed you, Son.
"""
    example1_ans = "yes"
    example2 = f"""
Oh, thanks for the answer. Here is another question.
Cartoon content: 
The scene is restaurant. 
Entities mentioned in the cartoon are: Alligator, Waiting Staff. 
An alligator is coming out of the floor. Two people stare at it. A waiter points at it. 
The uncanny point is that there is an alligator coming out of the floor.

Caption content: 
When the moon's in the sky like a big winking eye, that's emoji.
"""
    example2_ans = "no"
    example3 = f"""
Oh, thanks for the answer. Here is another question.
Cartoon content: 
The scene is a field. 
Entities mentioned in the cartoon are Fetch_(game), Dog, Anthropomorphism. 
Three dogs are out in the wild playing fetch. Two dogs are waiting patiently for the stick to be thrown. The third dog is standing on its hind legs and throwing the stick. 
The uncanny point is that A dog is standing on its hind legs and throwing a stick like a human would.

Caption content: 
He identifies with the oppressor.
"""
    example3_ans = "yes"
    example4 = f"""
Oh, thanks for the answer. Here is another question.
Cartoon content: 
The scene is a bedroom. 
Entities mentioned in the cartoon are Divorce, Lawyer, Television. 
There is a bedroom with a table in front of the bed. It looks like two routers are on the table. There are six people in the bed staring at it. 
The uncanny point is that There are a lot of people in one bed. They are also fully dressed.

Caption content: 
I've changed my mind. I love it.
"""
    example4_ans = "no"
    example5 = f"""
Oh, thanks for the answer. Here is another question.
Cartoon content: 
The scene is a living room. 
Entities mentioned in the cartoon are Giraffe, Television. 
Two giraffes are in a house. One is on the couch watching TV. 
The uncanny point is that Giraffes are doing human things.

Caption content: 
If he's elected, we're going to have a tough time proving we weren't born in Africa.
"""
    example5_ans = "yes"
    question = f"""
Oh, thanks for the answer. Here is another question.
Cartoon content: 
The scene is {scene}. 
Entities mentioned in the cartoon are {entities}. 
{image_description} 
The uncanny point is that {image_uncanny_description}

Caption content: 
{caption}
"""
    prompt_data = {
        'prompt': prompt, 
        'examples': [
            example1, 
            example2, 
            example3, 
            example4, 
            example5
        ], 
        'examples_ans': [
            example1_ans, 
            example2_ans, 
            example3_ans, 
            example4_ans, 
            example5_ans
        ], 
        'question': question
    }
    messages = [{'role': 'system', 'content': prompt_data['prompt']}]
    for example, example_ans in zip(prompt_data['examples'], prompt_data['examples_ans']):
        messages.append({'role': 'user', 'content': example})
        messages.append({'role': 'assistant', 'content': example_ans})
    messages.append({'role': 'user', 'content': prompt_data['question']})
    return messages