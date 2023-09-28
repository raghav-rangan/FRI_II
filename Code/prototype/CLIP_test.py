import os
import clip
import torch
import random
from PIL import Image
# from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
objs = ["laptop", "pc", "person", "desk", "table", "paper", "refrigerator", "phone", "lights", "wall"]
relats = ["on top of", "next to", "under", "above", "to the left of", "to the right of"]

obj_list = []
for obj1 in objs:
    for obj2 in objs:
        if obj1 == obj2:
            continue
        for relat in relats:
            obj_list.append(obj1 + " " + relat + " " + obj2)
for obj in objs:
    obj_list.append(obj + " is on the left side of the room")
    obj_list.append(obj + " is on the right side of the room")
    obj_list.append(obj + " is in the middle of the room")
    obj_list.append(obj + " is close to the viewer")
    obj_list.append(obj + " is far from the viewer")
    obj_list.append(obj + " is located higher in the room")
    obj_list.append(obj + " is located lower in the room")



# Prepare the inputs
image = Image.open("mock_kitchen.jpg").convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(c) for c in obj_list]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(len(obj_list))


pairs = set()

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    curr_str = obj_list[index]
    words = curr_str.split()
    obj1 = words[0]
    obj2 = words[len(words) - 1]
    if obj1 < obj2:
        temp = obj1
        obj1 = obj2
        obj2 = temp
    key = obj1 + "_" + obj2
    if key in pairs:
        continue
    pairs.add(key)
    # print(f"{obj_list[index]:>16s}: {100 * value.item():.2f}%")
    print(f"{obj_list[index]:>16s}")




#obj_list = ["monitors and pcs on desks", 
#             "pc sitting on desks and floor", 
#             "desks around the room", 
#             "person at the desks", 
#             "pillar in middle of room", 
#             "chair at desk", 
#             "whiteboard on wall", 
#             "google software", 
#             "dog running around", 
#             "apple on desk", 
#             "baseball", 
#             "keyboard on desk", 
#             "banana", 
#             "kitchen", 
#             "empty room", 
#             "person using computer",
#             "lunch",
#             "snacking",
#             "let him cook",
#             "person sitting on chair",
#             "person on table",
#             "person next to chair",
#             "person holding chair"]