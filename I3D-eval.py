from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import glob
from PIL import Image
import matplotlib.pyplot as plt
import time
import skimage.segmentation as seg
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import tensorflow as tf

from I3D_Pytorch import I3D

flag=0

_IMAGE_SIZE = 224  #shape=(1, 79, 224, 224, 3)

_SAMPLE_VIDEO_FRAMES = 79


#---------------INITIALIZE--------------------------

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
print("Using " , device , "...")

data_dir = '/Users/caglar/Desktop/Epic-Dataset'

annotations_dir = data_dir + "/annotations"
train_dir = data_dir + "/train"
#test_dir = data_dir + "/test"
val_dir = data_dir + "/validation"
df_train_action_labels = pd.read_csv(annotations_dir + '/EPIC_train_action_labels.csv')

class_paths_train = [d.path for d in os.scandir(train_dir) if d.is_dir]
whole_train_image_paths = []

path_len_train = 0
for c, class_path in enumerate(class_paths_train):
        paths = sorted(glob.glob(os.path.join(class_path, '*.jpg')))  
        whole_train_image_paths.extend(paths)
        path_len_train += len(paths)


class_paths_val = [d.path for d in os.scandir(val_dir) if d.is_dir]
whole_val_image_paths = []

path_len_val = 0
for c, class_path in enumerate(class_paths_val):
        paths = sorted(glob.glob(os.path.join(class_path, '*.jpg')))  
        whole_val_image_paths.extend(paths)
        path_len_val += len(paths)

_LABEL_MAP_PATH = 'data/label_map.txt'
_CHECKPOINT_PATHS ='data/pytorch_checkpoints/rgb_imagenet.pkl' #RGB MODEL

dataset_sizes = {}
dataset_sizes['train'] = int(path_len_train/200)
dataset_sizes['validation'] = int(path_len_val/200)

print(dataset_sizes)
#-------------------DATA LOADING---------------------------

class MySampler(torch.utils.data.Sampler):
    def __init__(self, path_len, seq_length):        
        indices = []
        
        for i in range(0, path_len, seq_length):
          if (i+seq_length < path_len):
            indices.append(torch.arange(i, i+1))
        
        
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        return iter(self.indices.tolist())
    
    def __len__(self):
        return len(self.indices)

class OwnDataset(Dataset):
    running_loss = 0.0
    running_corrects = 0
    
    def __init__(self, image_paths,transform, seq_length, length):
        self.image_paths = image_paths
        self.transform = transform
        self.seq_length = seq_length
        self.length = length
        
    def __getitem__(self, index):
        start_index = index
        stop_index = index + self.seq_length
        print("start_index: ", start_index , " stop_index: ", stop_index)

        increment_count = int(self.seq_length / intended_img_count)
        current_indices = random.sample(range(start_index, stop_index, increment_count), intended_img_count)
        current_indices = sorted( current_indices)


        first_image_index = current_indices[0]
        first_image_path_at_index = self.image_paths[current_indices[0]]
        
        video_name = first_image_path_at_index.split('/')[-2] # P01_01
        frame_name = first_image_path_at_index.split('/')[-1] # frame_0000000001.jpg
        frame_number = int(frame_name.split('_')[-1][:-4]) # 1 

        selected_row = df_train_action_labels.query(f"video_id == \"{video_name}\" and {frame_number}>= start_frame and {frame_number}<=stop_frame")
        selected_row = selected_row.drop(columns=['uid','participant_id','start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame', 'all_nouns', 'all_noun_classes'])
        row_list = []			
        for index, rows in selected_row.iterrows(): 
          my_list = rows.verb_class 
          row_list.append(my_list)

        image2 = Image.open(first_image_path_at_index) 
        
        if self.transform:
          image2 = self.transform(image2)
       
  
        seq_images = []
        for i in range(intended_img_count):
          current_img_path = current_indices[i]
          try:
            image = Image.open(self.image_paths[current_img_path]) 
          except:
            print("image couldn't loaded...")
            break

          if self.transform:
            image = self.transform(image)
          seq_images.append(image)
        

        seq_images = torch.cat(seq_images)
        seq_images = torch.reshape(seq_images, (3,intended_img_count, 224, 224))

        #seq_images = np_to_tensorflow(seq_images, dtype=np.float32)
        #seq_images = tf.reshape(seq_images , [1,intended_img_count, 224, 224,3] )

        x = seq_images
        y = row_list


        return x, y
    
    
    def __len__(self):
        return self.length



seq_length = 200
intended_img_count = 79

print("Train Image Path Count",  path_len_train)
print("Val Image Path Count",  path_len_val)

sampler_train = MySampler(path_len_train, seq_length)
sampler_val = MySampler(path_len_val, seq_length)

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset_train = OwnDataset(
    image_paths = whole_train_image_paths,
    transform = transform,
    seq_length=seq_length,
    length = len(whole_train_image_paths)
)
loader_train = DataLoader(
    dataset_train,
    batch_size=1,
    sampler = sampler_train,
    shuffle=False
)

dataset_val = OwnDataset(
    image_paths = whole_val_image_paths,
    transform = transform,
    seq_length = seq_length,
    length = len(whole_val_image_paths)
)
loader_val = DataLoader(
    dataset_val,
    batch_size=1,
    sampler = sampler_val,
    shuffle=False
)



#---------------I3D MODEL LOADING-------------------------------

NUM_CLASSES = 400
kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

 
rgb_i3d = I3D(num_classes=NUM_CLASSES , input_channel=3)

for param in rgb_i3d.features.parameters():
    param.requires_grad = True

state_dict = torch.load(_CHECKPOINT_PATHS)
rgb_i3d.load_state_dict(state_dict)

in_channels = rgb_i3d.features[18].in_channels
rgb_i3d.features[18] = nn.Conv3d(in_channels, 125, kernel_size=1, stride=1, bias=True)

print('RGB checkpoint restored')

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(rgb_i3d.features.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


#---------------------------TRAIN-----------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            
            dataloader = loader_train if (phase == 'train') else loader_val

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            for inputs, labels in dataloader:
                if (len(labels) != 0):
                    inputs = inputs.to(device)
                    labels = labels[0].to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        rbg_score, rgb_logits = rgb_i3d(inputs)
                        _, preds = torch.max(rgb_logits, 1)
                        #print("preds = ", preds)


                        loss = criterion(rgb_logits, labels)
                        #print("loss = ", loss)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model , "final_model.pt")
    return model


model_conv = train_model(rgb_i3d, criterion, optimizer,exp_lr_scheduler, num_epochs=2)




#--------------------------EVALUATE----------------------------

def eval_model(model, dataloader_param):
    total_size = 0
    top5_count = 0
    top1_count  = 0

    for i, data in enumerate(dataloader_param):
        inputs, labels = data
        
        if len(labels) != 0 :
            total_size += 1
            labels = labels[0]
            actual_verb_class = labels.item()

            print ("actual_verb_class" , actual_verb_class)
            print("my input shape : ", inputs.shape)

            rbg_score, rgb_logits = model(inputs)
            rgb_logits = rgb_logits[0]

            #print("rgb_logits shape", rgb_logits.shape)
            #print("rgb_logits : ", rgb_logits)

           
            out_predictions = F.softmax(rgb_logits)
            sorted_indices = np.argsort(out_predictions.data.numpy())[::-1]
            #print("sorted_indices", sorted_indices)

            rgb_logits = rgb_logits.data.numpy()
            out_predictions = out_predictions.data.numpy()

            print('\nNorm of logits: %f' % np.linalg.norm(rgb_logits))

            print('\nTop classes and probabilities')

            if( sorted_indices[:5][0] == actual_verb_class):
                top1_count += 1

            for index in sorted_indices[:5]:
                print(out_predictions[index], rgb_logits[index] , kinetics_classes[index])
                if ( index == actual_verb_class):
                    top5_count += 1


    top1_accuracy = top1_count/total_size * 100
    top5_accuracy = top5_count/total_size * 100
    print("I3D TOP1 ACCURACY = ", top1_accuracy)
    print("I3D TOP5 ACCURACY = ", top5_accuracy)

#------------------------------------------------------





