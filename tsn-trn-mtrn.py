
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
import skimage.segmentation as seg

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
from numpy import array

use_gpu = False
if(torch.cuda.is_available()):
  use_gpu = True

import torch.hub
repo = 'epic-kitchens/action-models'

class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'

tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model, 
                     pretrained='epic-kitchens', force_reload=True)

trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                    base_model=base_model, 
                    pretrained='epic-kitchens')

# tsm = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
#                     base_model=base_model, 
#                     pretrained='epic-kitchens')

mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                    base_model=base_model, 
                    pretrained='epic-kitchens')

data_dir = '/content/drive/My Drive/Videos'
annotations_dir = data_dir + "/annotations"
train_dir = data_dir + "/train"
test_dir = data_dir + "/test"

noun_dir = annotations_dir + '/noun.csv'

noun_file = open(noun_dir, 'r')
noun_file.readline()  # to skip first line that holds column names

noun_dict = {} 

for line in noun_file.readlines():
    line = line.rstrip('\n').split(',')
    key = int(line[0])                            
    value = [noun.replace('\'', '').replace('"', '').replace('[', '').replace(']', '').replace(' ', '') for noun in line[2:]]
    noun_dict[key] = value


verb_dir = annotations_dir + '/verb.csv'
verb_file = open(verb_dir, 'r')
verb_file.readline()  # to skip first line that holds column names

verb_dict = {} 

for line in verb_file.readlines():
    line = line.rstrip('\n').split(',')
    key = int(line[0])   # int verb id
    value = [noun.replace('\'', '').replace('"', '').replace('[', '').replace(']', '').replace(' ', '') for noun in line[2:]]
    verb_dict[key] = value


class MySampler(torch.utils.data.Sampler):
    def __init__(self, path_len, seq_length):        
        indices = []
        
        for i in range(0, path_len, seq_length):
          if (i+seq_length < path_len):
            indices.append(torch.arange(i, i+1))
        
        
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
#         indices = self.indices[torch.randperm(len(self.indices))]
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

        current_indices = torch.randint(start_index, stop_index, (intended_img_count,))
        current_indices , indices = torch.sort( current_indices)


        first_image_index = current_indices[0]
        first_image_path_at_index = self.image_paths[current_indices[0]]
        
        video_name = first_image_path_at_index.split('/')[-2] # P01_01
        frame_name = first_image_path_at_index.split('/')[-1] # frame_0000000001.jpg
        frame_number = int(frame_name.split('_')[-1][:-4]) # 1 

        selected_row = df_train_action_labels.query(f"video_id == \"{video_name}\" and {frame_number}>= start_frame and {frame_number}<=stop_frame")
        selected_row = selected_row.drop(columns=['uid','participant_id','start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame', 'all_nouns', 'all_noun_classes'])
        row_list = []			
        for index, rows in selected_row.iterrows(): 
          my_list =[rows.video_id ,rows.verb, rows.verb_class, rows.noun, rows.noun_class] 
          row_list.append(my_list)

       image2 = Image.open(first_image_path_at_index) 
        
        if self.transform:
          image2 = self.transform(image2)
#             plt.imshow(image)
#             plt.show()
       
  
        seq_images = []
        for i in range(intended_img_count):
          current_img_path = current_indices[i]
          print('Getting images from {} '.format(current_indices[i]))
          try:
            image = Image.open(self.image_paths[current_img_path]) 
          except:
            print("image couldn't loaded...")
            break

          if self.transform:
            image = self.transform(image)
          seq_images.append(image)

        seq_images = torch.cat(seq_images)
        seq_images = torch.reshape(seq_images, (1,intended_img_count, 224, 224,3))

        x = seq_images
        y = row_list
        print("seq_images shape: ",seq_images.shape)
        # z = torch.tensor([[ float(i) for i in bbox2 ]], dtype=torch.float)


        return x, y
    
    
    def __len__(self):
        return self.length
      
      

df_train_action_labels = pd.read_csv(annotations_dir + '/EPIC_train_action_labels.csv')

class_paths_train = [d.path for d in os.scandir(train_dir) if d.is_dir]
whole_train_image_paths = []

train_video_names = [path.split('/')[-1] for path in class_paths_train]
# print(train_video_names) g

path_len = 0
for c, class_path in enumerate(class_paths_train):
        paths = sorted(glob.glob(os.path.join(class_path, '*.jpg')))  
        whole_train_image_paths.extend(paths)
        path_len += len(paths)

seq_length = 200
intended_img_count = 8

print(path_len)

sampler = MySampler(path_len, seq_length)

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
    sampler = sampler,
    shuffle=False
)

from google.colab import drive
drive.mount('/content/drive')
success_verb_tsn = 0
success_verb_tsn_top5 = 0
success_noun_tsn = 0
success_noun_tsn_top5 = 0
success_verb_trn = 0
success_verb_trn_top5 = 0
success_noun_trn = 0
success_noun_trn_top5 = 0
success_verb_mtrn = 0
success_verb_mtrn_top5 = 0
success_noun_mtrn = 0
success_noun_mtrn_top5 = 0

model_tsn = tsn
model_trn = trn
model_mtrn = mtrn

total_size = 0
predicted_noun = []
actual_noun = []

predicted_verb = []
actual_verb = []

dictionary_noun = {}
dictionary_verb = {}
classes = []
classes_unique = []
for k in verb_dict:
  classes.append(verb_dict[k][0])

for i, data in enumerate(loader_train):
  total_size+=1
  # labels => (video_id), (verb), (verb_class), (noun), (noun_class) 
      # to get video id -> labels[0][0][0]
  # inputs => torch.Size([1, 24, 224, 224])
  
  inputs,labels = data
  actual_verb_class_id = -1
  actual_noun_class_id = 0
  
  actual_verb_label = ""
  
  if (len(labels) != 0):
    actual_verb_class_id = labels[0][2].item()
    actual_noun_class_id = labels[0][4].item()
    actual_verb_label = labels[0][1][0]
  print(actual_verb_label)
  
  #-------------------TSN--------------------------
  model_tsn.eval()
  model_tsn.requires_grad_(requires_grad=False)
  output_tsn = model_tsn.features(inputs)

  verb_logits_tsn, noun_logits_tsn = model_tsn.logits(output_tsn)

  res_tsn_verb, ind_tsn_verb = verb_logits_tsn.topk(5)
  flag=0

  for indice in ind_tsn_verb[0]:
    if indice.item() == actual_verb_class_id:
      success_verb_tsn_top5 += 1
      print(indice)
      predicted_verb.append(indice.item())
      flag=1

  if(flag == 0):
    for indice in ind_tsn_verb[0]:
      predicted_verb.append(indice.item())
      print(indice)
      break
  actual_verb.append(actual_verb_class_id)

  res_tsn_noun, ind_tsn_noun = noun_logits_tsn.topk(5)

  for indice in ind_tsn_noun[0]:
    if indice.item() == actual_noun_class_id:
      success_noun_tsn_top5 += 1

  verb_value_tsn, verb_indice_tsn = torch.max(verb_logits_tsn, 1)
  noun_value_tsn, noun_indice_tsn = torch.max(noun_logits_tsn, 1)

  predicted_verb_class_id_tsn = verb_indice_tsn[0].item()
  predicted_noun_class_id_tsn = noun_indice_tsn[0].item()

  predicted_verb_tsn = verb_dict[predicted_verb_class_id_tsn]
   
  predicted_noun_tsn = noun_dict[predicted_noun_class_id_tsn]

  print(actual_verb_class_id)
  if predicted_verb_class_id_tsn == actual_verb_class_id:
    success_verb_tsn += 1

  if predicted_noun_class_id_tsn == actual_noun_class_id:
    success_noun_tsn += 1



  #-------------------TRN--------------------------

  model_trn.eval()
  model_trn.requires_grad_(requires_grad=False)
  output_trn = model_trn.features(inputs)

  verb_logits_trn, noun_logits_trn = model_trn.logits(output_trn)

  res_trn_verb, ind_trn_verb = verb_logits_trn.topk(5)

  for indice in ind_trn_verb[0]:
    if indice.item() == actual_verb_class_id:
      success_verb_trn_top5 += 1

  res_trn_noun, ind_trn_noun = noun_logits_trn.topk(5)
  for indice in ind_trn_noun[0]:
    if indice.item() == actual_noun_class_id:
      success_noun_trn_top5 += 1
 
  verb_value_trn, verb_indice_trn = torch.max(verb_logits_trn, 1)
  noun_value_trn, noun_indice_trn = torch.max(noun_logits_trn, 1)

  
  predicted_verb_class_id_trn = verb_indice_trn[0].item()
  predicted_noun_class_id_trn = noun_indice_trn[0].item()

  predicted_verb_trn = verb_dict[predicted_verb_class_id_trn]
  predicted_noun_trn = noun_dict[predicted_noun_class_id_trn]

 
  if predicted_verb_class_id_trn == actual_verb_class_id:
    success_verb_trn += 1

  if predicted_noun_class_id_trn == actual_noun_class_id:
    success_noun_trn += 1

  #-------------------MTRN--------------------------

  model_mtrn.eval()
  model_mtrn.requires_grad_(requires_grad=False)
  output_mtrn = model_mtrn.features(inputs)

  verb_logits_mtrn, noun_logits_mtrn = model_mtrn.logits(output_mtrn)

  res_mtrn_verb, ind_mtrn_verb = verb_logits_mtrn.topk(5)

  for indice in ind_mtrn_verb[0]:
    if indice.item() == actual_verb_class_id:
      success_verb_mtrn_top5 += 1

  res_mtrn_noun, ind_mtrn_noun = noun_logits_mtrn.topk(5)
  for indice in ind_mtrn_noun[0]:
    if indice.item() == actual_noun_class_id:
      success_noun_mtrn_top5 += 1

  verb_value_mtrn, verb_indice_mtrn = torch.max(verb_logits_mtrn, 1)
  noun_value_mtrn, noun_indice_mtrn= torch.max(noun_logits_mtrn, 1)

  predicted_verb_class_id_mtrn = verb_indice_mtrn[0].item()
  predicted_noun_class_id_mtrn = noun_indice_mtrn[0].item()

  predicted_verb_mtrn = verb_dict[predicted_verb_class_id_mtrn]
  predicted_noun_mtrn = noun_dict[predicted_noun_class_id_mtrn]

 
  if predicted_verb_class_id_mtrn == actual_verb_class_id:
    success_verb_mtrn += 1

  if predicted_noun_class_id_mtrn == actual_noun_class_id:
    success_noun_mtrn += 1

  #----------------------------------------------

  print(8*i)
  print("Actual verb class id: ", actual_verb_class_id)
  print("Predicted verb class id: ", predicted_verb_class_id_tsn)
  print("---------------------------------")
  print("Actual noun class id: ", actual_noun_class_id)
  print("Predicted noun class id: ", predicted_noun_class_id_tsn)
  print("---------------------------------")
  predicted_verb_indice = torch.argmax(verb_logits[0][0])
  predicted_noun_indice = torch.argmax(noun_logits)
  
  print("verb indice is:", predicted_verb,"i: ", 8*i)
  print(verb_logits) #" **** ", noun_logits.shape)
  print(noun_logits)
if isinstance(output, tuple):
     print(f"Outputsssss shape: {[o.shape for o in output]}")
 else:
     print(f"Output shape: {output.shape}")

for i in range(len(classes)):
  predicted_verb.append(i)
  actual_verb.append(i)


print(predicted_verb)
cm = confusion_matrix(actual_verb, predicted_verb)
fig, ax = plt.subplots(1, figsize=(30, 30))
im = ax.imshow(cm, interpolation='nearest')


sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix, without normalization',
           ylabel='True label',
           xlabel='Predicted label')
fig.tight_layout()
fig.savefig('temp.png')
plt.show()


print("TSN Success verb predict:", success_verb_tsn)
print("TSN Accuracy for verb: %" , success_verb_tsn/total_size *100)
print("TSN Success noun predict: ", success_noun_tsn)
print("TSN Accuracy for noun: %" , success_noun_tsn/total_size *100)
print("TOP5-TSN Accuracy for verb: %", success_verb_tsn_top5 / total_size * 100)
print("TOP5-TSN Accuracy for noun: %" , success_noun_tsn_top5 / total_size * 100)

print("TRN Success verb predict:", success_verb_trn)
print("TRN Accuracy for verb: %" , success_verb_trn/total_size *100)
print("TRN Success noun predict: ", success_noun_trn)
print("TRN Accuracy for noun: %" , success_noun_trn/total_size *100)
print("TOP5-TRN Accuracy for verb: %", success_verb_trn_top5 / total_size * 100)
print("TOP5-TRN Accuracy for noun: %" , success_noun_trn_top5 / total_size * 100)

print("MTRN Success verb predict:", success_verb_mtrn)
print("MTRN Accuracy for verb: %" , success_verb_mtrn/total_size *100)
print("MTRN Success noun predict: ", success_noun_mtrn)
print("MTRN Accuracy for noun: %" , success_noun_mtrn/total_size *100)
print("TOP5-MTRN Accuracy for verb: %", success_verb_mtrn_top5 / total_size * 100)
print("TOP5-MTRN Accuracy for noun: %" , success_noun_mtrn_top5 / total_size * 100)