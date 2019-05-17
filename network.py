'''
-------------------------------------------------------------
network.py
-------------------------------------------------------------
File contains:
    Important: images are assumed to be 3x48x48 tensors 
    Part of network class:
    - Network architecture:
        - 3 convolution layers and 3 pooling layers: (3x48x48 -> 32x24x24 -> 64x12x12 -> 128x6x6)
        - 2 fully connected layers: (128*6*6 : average between # of inputs and # outputs -> average : # of classes)
        - Dropout p = 0.5
    
    - Forward function 
        - ReLU activation
        
    Seperate from class:
    - train_model function
        - Function trains the model and validates it against a validation set
        - Prints # of epochs, train_loss, valid_loss, accuracy on validation set
        - Saves data to kanto_starter_classifier.pt if valid_loss is less than the minimum valid loss
        - Returns TRUE if successfully run
        
    - load_model function
        - Just reminds user to load using pytorch 
-------------------------------------------------------------
'''
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Constants
        self.im_len = 48
        self.im_wid = 48
        self.im_dep = 3
        self.depth_1 = 32
        self.depth_2 = self.depth_1 * 2
        self.depth_3 = self.depth_2 * 2
        self.n_output = 5 #Number of classes
        self.fc_input = int(self.depth_3 * (self.im_len / (2**3)) * (self.im_wid / (2**3))) 
        #Above: 2 = pooling reduction to im wid/len, 3 = num times pooling layer is used. TODO: use meaningul names
        self.hidden_size = int((self.fc_input + self.n_output)/2)
        
        #Convolutional layers
        self.conv1_1 = nn.Conv2d(self.im_dep, self.depth_1, 3, padding = 1)
        self.conv2_1 = nn.Conv2d(self.depth_1, self.depth_2, 3, padding = 1)
        self.conv3_1 = nn.Conv2d(self.depth_2, self.depth_3, 3, padding = 1)

        
        #Fully connected layers 
        self.fc1 = nn.Linear(self.fc_input, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.n_output)
        
        #Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        #Dropout probability 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        #Passing image through convolution layers
        x = F.relu(self.conv1_1(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = self.pool(x)

        #Flattening image
        x = x.view(-1, self.fc_input)
        
        #Passing image through fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

def train_model(model, n_epochs, train_loader, valid_loader, cuda_avaliability):
    file_save_to = 'kanto_starter_classifier.pt'
    if model == None:
        return False #Error case
    
    if cuda_avaliability == True:
        model = model.cuda()
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.03)
    min_valid_loss = 99999 #Arbiturary big number
    
    #Training
    for e in range(0, n_epochs, 1):
        train_loss = 0  
        valid_loss = 0
        accuracy = 0
        model.train()
        for images, labels in train_loader:
            if cuda_avaliability == True:
                images = images.cuda()
                labels = labels.cuda()
                
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss.item()
          
        #Validating
        model.eval()
        for images, labels in valid_loader:
            if cuda_avaliability == True:
                images = images.cuda()
                labels = labels.cuda()
        
            output = model.forward(images)
            loss = criterion(output, labels)
        
            #Measuring validation loss
            valid_loss = valid_loss + loss.item()
            
            #Measuring accuracy
            top_prob, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view((*top_class.shape))
            if cuda_avaliability == True:
                accuracy = accuracy + torch.mean(equals.type(torch.cuda.FloatTensor))
            else:
                accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor))
        
        #Computing averages and printing                    
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        print('\n.... EPOCH #', e, ' ....')
        print('train_loss = ', train_loss)
        print('valid_loss = ', valid_loss)
        print('test accuracy = ', accuracy.item()/len(valid_loader))
        
        #Saving if current valid loss is less than the min valid loss ever obtained
        if valid_loss <= min_valid_loss:
            print('Saving model...')
            torch.save(model.state_dict(), file_save_to)
            min_valid_loss = valid_loss
            
    return True

def load_model(model):
    print('Load model using PyTorch functions. Not done in this function')
