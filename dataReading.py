import numpy as np
import pandas
from PIL import Image

# data = np.genfromtxt('.Data_Entry_2017_v2020.csv.icloud', delimiter=',')
data = pandas.read_csv('Data_Entry_2017_v2020.csv')
print(data)
imageNames = data[data.columns[0]]
classification = data[data.columns[1]]
print(imageNames)
dataset = []


for i in range(10):
    image = Image.open(imageNames[i])
    image_resized = image.resize((64,64))
    image_resized.show()
    rawDiagnosis = classification[i]
    diagnosisSplit = rawDiagnosis.split("|")
    diagnosis = diagnosisSplit[0]
    print("it ran")
    print(diagnosis)
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
    
    print(num)
