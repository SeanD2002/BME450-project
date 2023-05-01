import numpy as np
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#code to load data from csv file and folder
data = []
imageNames = []
classification = []
i = 0
included_cols = [0,1]
with open('Data_Entry_2017_v2020.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',', quotechar='|')
    for row in reader:
        content = list(row[i] for i in included_cols)
        imageNames.insert(i, content[0])
        classification.insert(i,content[1])
        i = i+1

images = []
diagnoses = []
dataset = []

i = 0
backSlash = "C:\\Users\\sadev\\images" + "\\"
#example =  Image.open("C:\\Users\\sadev\\images" + "\\" + imageNames[1])
#example.show()
#example = example.resize((64,64))
#example.show()

#code to iterate through 100,000 values and use only 5000
for i in range(1,5000):
    image = Image.open(backSlash + imageNames[i])
    image_resized = image.resize((64,64))
    rawDiagnosis = classification[i]
    diagnosisSplit = rawDiagnosis.split("|")
    diagnosis = diagnosisSplit[0]
    if diagnosis == "Pneumothorax":
        num = 0
    elif diagnosis == "Pneumonia":
        num = 1
    elif diagnosis == "Pleural_Thickening":
        num = 2
    elif diagnosis == "Nodule": 
        num = 3
    elif diagnosis == "Mass":
        num = 4
    elif diagnosis == "Infiltration":
        num = 5
    elif diagnosis == "Hernia":
        num = 6
    elif diagnosis == "Fibrosis":
        num = 7
    elif diagnosis == "Emphysema":
        num = 8
    elif diagnosis == "Effusion":
        num = 9
    elif diagnosis == "Edema":
        num = 10
    elif diagnosis == "Consolidation":
        num = 11
    elif diagnosis == "Cardiomegaly":
        num = 12
    elif diagnosis == "Atelectasis":
        num = 13
    elif diagnosis == "No Finding":
        num = 14
    else:
        print(diagnosis)
    diagnoses.append(num)
    dataTuple = (image_resized, num)
    dataset.append(dataTuple)   



# Data loading
class MyTransform(object):
    def __init__(self, size=(64, 64), mean=0.5, std=0.5):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = img.resize(self.size)
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = torch.from_numpy(img).unsqueeze(0)
        return img
class MyDataset(Dataset):
    def __init__(self, root_dir, image_filenames, labels, transform=None):
        self.root_dir = root_dir
        self.image_filenames = image_filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_filenames[index])
        img = Image.open(img_path).convert('L')
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label

#creates class to construct nerual network
class Net(nn.Module):
    def __init__(self, l1_reg = 0.01, l2_reg = 0.01):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 15)
        self.dropout = nn.Dropout(p = 0.5)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    #attempt to add L1/l2 regularization to neural network
    """def l1_penalty(self):
        l1 = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1 += torch.norm(param, 1)
        return l1
    
    def l2_penalty(self):
        l2 = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2 += torch.norm(param, 2)
        return l2
    
    def loss_fn(self, outputs, labels, l1_lambda=0.001, l2_lambda=0.0001):
        criterion = nn.CrossEntropyLoss()
        l1 = self.l1_penalty()
        l2 = self.l2_penalty()
        loss = criterion(outputs, labels) + l1_lambda * l1 + l2_lambda * l2
        return loss"""

#function to train our model    
def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0.0
  
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    return train_loss / len(dataloader.dataset)

#function to test our model
def test(model, dataloader):
    model.eval()
    test_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    return test_loss / len(dataloader.dataset), accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#sets up variables to run with code
lr = 0.001
num_epochs = 30
batch_size = 32
root_dir = backSlash
image_filenames = imageNames[1:5000] 
labels = diagnoses[0:5000] 
inter = np.array(labels)
labels = torch.from_numpy(inter)
labels = labels.long()

transform = MyTransform()
dataset = MyDataset(root_dir, image_filenames, labels, transform=transform)


test_split = 0.2
test_size = int(len(dataset) * test_split)
train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

model = Net().to(device)


criterion = nn.CrossEntropyLoss()
#more code in attemot to add l1/l2 regularization
#l1_lambda = 0.001
#l2_lambda = 0.0001
#criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]), reduction='mean')
#l1_reg = nn.L1Loss(reduction='mean')
#l2_reg = nn.MSELoss(reduction='mean')
#for param in model.parameters():
#        criterion = criterion + l1_lambda * l1_reg(param) + l2_lambda * l2_reg(param)

optimizer = optim.Adam(model.parameters(), lr=lr)

#sets up data for use from tensors
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

roundNumber = []
trainLoss = []
testLoss = []

#iterate through number of epochs to train and test dataset
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss, accuracy = test(model, test_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    roundNumber.append(epoch+1)
    trainLoss.append(train_loss)
    testLoss.append(test_loss)
#plots test and train loss from epochs
plt.plot(roundNumber,trainLoss, label = "train")
plt.plot(roundNumber,testLoss, label = 'test')
plt.legend()
plt.show()
print('finish')


