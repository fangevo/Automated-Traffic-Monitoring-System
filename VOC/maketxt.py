# Splitting into training and validation sets and generating txt files of dataset image IDs in the txt_dataset folder.
import os
import random

trainval_percent = 1.0  # All data for training + validation
train_percent = 0.8     # 80% for training set, 20% for validation set

xmlfilepath = 'VOC/xml_dataset'
txtsavepath = 'VOC/txt_dataset'

total_xml = os.listdir(xmlfilepath)
num = len(total_xml)

list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)

trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('VOC/txt_dataset/trainval.txt', 'w')
ftrain = open('VOC/txt_dataset/train.txt', 'w')
fval = open('VOC/txt_dataset/val.txt', 'w')

# Divide the data set
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)

ftrainval.close()
ftrain.close()
fval.close()

print('The training set and validation set division is complete! three txt files containing the image IDs has been saved.')