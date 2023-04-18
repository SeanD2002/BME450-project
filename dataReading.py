import numpy as np
import pandas
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# data = np.genfromtxt('.Data_Entry_2017_v2020.csv.icloud', delimiter=',')
data = pandas.read_csv('Data_Entry_2017_v2020.csv')
print(data)
imageNames = data[data.columns[0]]
classification = data[data.columns[1]]
print(imageNames)
images = []
diagnoses = []

#
for i in range(10):
    image = Image.open(imageNames[i])
    image_resized = image.resize((64,64))
    images.append(image_resized)
    #image_resized.show()
    rawDiagnosis = classification[i]
    diagnosisSplit = rawDiagnosis.split("|")
    diagnosis = diagnosisSplit[0]
    #print(diagnosis)
    if diagnosis == "Pneumothorax":
        num = 1
    elif diagnosis == "Pneuomonia":
        num = 2
    elif diagnosis == "Pleural Thickening":
        num = 3
    elif diagnosis == "Nodule": 
        num = 4
    elif diagnosis == "Mass":
        num = 5
    elif diagnosis == "Infiltration":
        num = 6
    elif diagnosis == "Hernia":
        num = 7
    elif diagnosis == "Fibrosis":
        num = 8
    elif diagnosis == "Emphysema":
        num = 9
    elif diagnosis == "Effusion":
        num = 10
    elif diagnosis == "Edema":
        num = 11
    elif diagnosis == "Consolidation":
        num = 12
    elif diagnosis == "Cardiomegaly":
        num = 13
    elif diagnosis == "Atelactasis":
        num = 14
    elif diagnosis == "No Finding":
        num = 15
    
    diagnoses.append(num)
    
class ConvNet(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Data loading
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = ConvNet(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the {len(test_loader.dataset)} test images: {100 * correct / total}%')

# Save the model checkpoint
torch.save(model.state_dict(), 'convnet_mnist.ckpt')
