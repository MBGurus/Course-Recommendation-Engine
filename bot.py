import random

import json

import torch

from model import Chat

from nltk_utils import words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as b:
    intents = json.load(b)

FILE = "data.pth"   
data = torch.load(FILE,weights_only=True) 

input_size =data["input_size"]
hidden_size =data["hidden_size"]
output_size =data["output_size"]
all_words =data["all_words"]
tags =data["tags"]
model_state =data["model_state"]

model = Chat(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ZEN"
print("let's chat type 'quit' to exit")

while True:
    sentence = input('you: ')
    if sentence =="quit:":
        break
    sentence = tokenize(sentence)
    x = words(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1) #put zero if not working
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand......") 