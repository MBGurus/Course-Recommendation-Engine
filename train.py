import json
from nltk_utils import tokenize, stem ,words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import Chat


with open('intents.json','r') as f:
    intents = json.load(f)


all_words=[]
tags =[]
xy=[]

for intent in intents['intents']:
       tag= intent['tag']
       tags.append(tag)
       for pattern in intent['patterns']:
          w = tokenize(pattern)   
          all_words.extend(w)
          xy.append((w,tag))
    
ignore_words = ["?","!",".",","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


x_data = []
y_data = []
for (pattern_line,tag) in xy:
     bag = words(pattern_line,all_words)
     x_data.append(bag)

     label = tags.index(tag)
     y_data.append(label) 
x_data = np.array(x_data)
y_data = np.array(y_data)

class ChatDataset(Dataset):
     def __init__(self):
          self.n_samples = len(x_data)
          self.x_dat = x_data
          self.y_dat = y_data
    #Dataset[idx]
     def __getitem__(self, index):
          return self.x_dat[index] , self.y_dat[index]
     
     def __len__(self):
          return self.n_samples
     
#Hyperparameters  
batch_size =  8  
input_size = len(x_data[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Chat(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate )

for epoch in range(num_epochs):
    for (word,labels) in train_loader:
          word = word.to(device)
          labels = labels.to(device)

          outputs = model(word)
          loss = criterion(outputs,labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    if (epoch +1) % 100==0:
        print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
     "model_state": model.state_dict(),
     "input_size": input_size,
     "output_size":output_size,
     "hidden_size":hidden_size,
     "all_words": all_words,
     "tags":tags
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'training complete. file saved {FILE}')