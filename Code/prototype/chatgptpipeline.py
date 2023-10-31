import json
import random
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import os
import openai
from dotenv import load_dotenv


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)
        
def print_list(name, input_list, scores=None):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i]))
    
def draw_image(img_path, boxes, box_labels, rel_labels, box_scores=None, rel_scores=None):
    size = get_size(Image.open(img_path).size)
    pic = Image.open(img_path).resize(size)
    num_obj = len(boxes)
    for i in range(num_obj):
        info = str(i) + '_' + box_labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    display(pic)
    print('*' * 50)
    print_list('box_labels', box_labels, box_scores)
    print('*' * 50)
    print_list('rel_labels', rel_labels, rel_scores)
    
    return None

def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)

def generate_response(question, context):
    # Define the context containing the bank of phrases. Need to store each relation twice to allow for querying in both directions
    openai.api_key = api_key
    prompt = f"{question}\nContext: {context}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text

def generate_context(prompt, triplets, rel_topk, obj_labels):
    words = prompt.split()
    objs_seen = set()
    for word in words:
        if word in obj_labels:
            objs_seen.add(word)
        if word[:-1] in obj_labels:
            objs_seen.add(word[:-1])
    context = ''
    for i in range(rel_topk):
        if triplets[i][0] in objs_seen or triplets[i][2] in objs_seen:
            context += '(' + triplets[i][0] + ' ' + triplets[i][1] + ' ' + triplets[i][2] + '), '
    return context
    


# load the following to files from DETECTED_SGG_DIR
custom_prediction = json.load(open('/Users/kevin/FRI/SGG visualization/custom_prediction.json'))
custom_data_info = json.load(open('/Users/kevin/FRI/SGG visualization/custom_data_info.json'))

# parameters
image_idx = 6
box_topk = 30 # select top k bounding boxes
rel_topk = 100 # select top k relationships
ind_to_classes = custom_data_info['ind_to_classes']
ind_to_predicates = custom_data_info['ind_to_predicates']
image_path = custom_data_info['idx_to_files'][image_idx]
boxes = custom_prediction[str(image_idx)]['bbox'][:box_topk]
box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
box_scores = custom_prediction[str(image_idx)]['bbox_scores'][:box_topk]
all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

for i in range(len(box_labels)):
    box_labels[i] = ind_to_classes[box_labels[i]]
box_labels = box_labels[:box_topk]

rel_labels = []
rel_scores = []
triplets = []
for i in range(len(all_rel_pairs)):
    if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
        rel_scores.append(all_rel_scores[i])
        label = box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + box_labels[all_rel_pairs[i][1]]
        triplets.append((box_labels[all_rel_pairs[i][0]], ind_to_predicates[all_rel_labels[i]], box_labels[all_rel_pairs[i][1]]))
        rel_labels.append(label)

rel_labels = rel_labels[:rel_topk]
rel_scores = rel_scores[:rel_topk]

# print(image_path)
# print("\nObjects:")

# for i in range(box_topk):
#     print(box_labels[i])

# print("\nRelationships:")

# for i in range(rel_topk):
#     print(rel_labels[i])

# print("\n\n\n\n")


context = ''
for i in range(rel_topk):
    context += '(' + rel_labels[i] + '), '

load_dotenv()

preinstructions = 'Given the following list relationships between objects as determined by an AI model, \
                please answer my questions as you would a human interpreting the data. However, make \
                no references to the data itself, and act as if you are an observer in this \
                room. Here is the data:'

postinstructions = 'Do not mention the data or existence of the data at all, and respond as if you were \
                    a normal person viewing the room. Use only the information listed here. DO NOT make assumptions \
                    about other objects in the room.'

# Your OpenAI API key
api_key=os.getenv('OPENAI_API_KEY')


print("Welcome to the Object Position Query System")
while True:
    user_input = input("Ask a question (e.g., 'Where is the chair in relation to the couch?') or type 'exit' to quit: ")

    if user_input.lower() == 'exit':
        break

    # Use the user's question as a prompt to the model
    prompt = user_input.strip() + "\n"
    cur_context = generate_context(prompt, triplets, rel_topk, box_labels)
    if len(cur_context) == 0:
        cur_context = context
    print(cur_context)
    response = generate_response(prompt, preinstructions + '\n' + cur_context + '\n' + postinstructions)

    print("Response: " + response)
    print("\n\n")
