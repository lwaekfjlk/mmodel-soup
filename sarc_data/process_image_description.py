import ast

with open("intermediate_data/sarc_image_description.txt", 'r') as f:
    data = f.readlines()

for line in data:
    image_id, image_description = line.strip().split(" ", 1)
    image_description = ast.literal_eval(image_description)
    print(image_id, image_description)