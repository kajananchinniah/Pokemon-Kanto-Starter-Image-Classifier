'''
-------------------------------------------------------------
train.py
-------------------------------------------------------------
Script that does the following:
    - Intialize some constants for training
    - Apply transforms
    - Loads dataset
    - Splits set into a training set and validation set
    - Obtains iterators
    - Creates a variable containing a list of all classes (in terms of name)
    - initializes model and prints out the architecture
    - Trains the model
-------------------------------------------------------------

'''
import network
import torch
from torchvision import datasets, transforms
import numpy
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path

#Initalizing constants
data_dir = 'starter_dataset/'
num_workers = 0
batch_size = 32
validation_percent = 0.2
n_epochs = 30
cuda_avaliability = torch.cuda.is_available()

#Applying transformations onto set
transform = transforms.Compose([transforms.Resize(48), transforms.CenterCrop(48),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], 
                                                     [0.5, 0.5, 0.5])])

#Loading image dataset
dataset = datasets.ImageFolder(data_dir, transform = transform)

#Creating split between training set and validation set 
num_train = len(dataset)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(validation_percent * num_train))
train_ind = indices[split:]
valid_ind = indices[:split]
train_sampler = SubsetRandomSampler(train_ind)
valid_sampler = SubsetRandomSampler(valid_ind)

#Obtaining iterators 
train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                           sampler = valid_sampler, num_workers = num_workers)

#Creating list of classes
classes = []
p = Path('starter_dataset/')
dirs = p.glob('*')
for folder in dirs:
    label = str(folder).split('/')[-1]
    label = label.replace("starter_dataset\\", "")
    classes.append(label)
        
#Initializing network and printing architecture
model = network.network()
print(model)

#Training model. Note: saving is done within the function
network.train_model(model, n_epochs, train_loader, valid_loader, cuda_avaliability)