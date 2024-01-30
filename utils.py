import os
import sys
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm

def read_split_data(root:str,val_rate:float=0.3, plot = True):
    random.seed(0)
    class_list = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root,cls))] #each folder represent one class
    class_list.sort()
    class_idx = dict((k,v) for v,k in enumerate(class_list)) #generate indexs of every class

    train_data = [] #record paths for training_datas
    train_label = [] #record paths for training_labels
    val_data = [] #record paths for validation_datas
    val_label = [] #record paths for validation_labels
    every_class_num = [] #record number for each class
    supported_ext = [".jpg",".JPG",".png",".PNG",".jpeg",".JPEG"]
    # traverse each folder
    for cls in class_list:
        cls_path = os.path.join(root,cls)
        images = [os.path.join(root,cls,i) for i in os.listdir(cls_path) if os.path.splitext(i)[-1] in supported_ext]
        images.sort()
        image_class = class_idx[cls]
        every_class_num.append(len(images))
        val_set = random.sample(images, k=int(len(images)*val_rate))
        # traverse every image
        for img in images:
            if img in val_set:
                val_data.append(img)
                val_label.append(image_class)
            else:
                train_data.append(img)
                train_label.append(image_class)
    print(f"{sum(every_class_num)} images were found. {len(train_data)} for training and {len(val_data)} for validation.")
    # draw the class distribution
    if plot:
        plt.bar(range(len(class_list)), every_class_num, align = 'center', color='g')
        plt.title("class distribution")
        plt.show()
    
    return train_data,train_label,val_data,val_label

def train_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device) #accumulate loss
    accu_num = torch.zeros(1).to(device) #accumulate correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)
    for stp, data in enumerate(data_loader):
        img, labels = data
        sample_num += img.shape[0]
        pred = model(img.to(device))
        pred_classes = torch.max(pred,dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred,labels.to(device))
        loss.backward()
        accu_loss+=loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (stp + 1),
                                                                               accu_num.item() / sample_num)

        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item()/(stp+1), accu_num.item()/sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)

    for stp,data in enumerate(data_loader):
        imgs,labels  = data
        sample_num += imgs.shape[0]
        pred = model(imgs.to(device))
        pred_classes = torch.max(pred,dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred,labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (stp + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (stp + 1), accu_num.item() / sample_num
    