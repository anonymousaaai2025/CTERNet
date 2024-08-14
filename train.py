from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.optim as optim
from torch.optim import lr_scheduler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from vit import ViT
# from models.model.WSCNet01031 import ResNetWSL, ClassWisePool
from models.CTERNet import ResNetWSL, ClassWisePool
# from models.model.CauMol.M1201CAM import ResNetWSL, ClassWisePool
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from lib.CAM import CAM_Module
from lib.sobel import Sobel
from torch.autograd import Variable

import matplotlib
from matplotlib import cm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import os
import copy
import itertools

import warnings

warnings.filterwarnings('ignore')

from PIL import ImageFile, Image

# print(torch.cuda.is_available())
# print(torch.version.cuda)


ImageFile.LOAD_TRUNCATED_IMAGES = True
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.backends.cudnn.enabled = False
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = './FI/'
# data_dir = '/home/dell/newStuFile/zxy/yolov5/data/images/Emotion-6'
# data_dir = '/home/dell/newStuFile/zxy/WSCNet/twitter/2'
# train:19292 val:3408 total:22700

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 8

# Number of feature map in each class
num_maps = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 18

# Number of epochs to train forNameError: name 'itertools' is not defined
num_epochs = 30

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

model_savepath = './models/'

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # drawConfu(model)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and valid-13ation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_dec = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # print('inputs shape is ', inputs.shape)
                labels = labels.to(device)
                # print('label is ', labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with (((torch.set_grad_enabled(phase == 'train')))):
                    get_gradient = Sobel().to(device)
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)

                        loss = criterion(outputs, labels) + \
                               criterion(aux_outputs, labels)
                    else:
                        outputs, aux_outputs = model(inputs)

                        loss = criterion(outputs, labels) + criterion(aux_outputs,labels)

                    _, preds = torch.max(aux_outputs, 1)
                    _, preds_dec = torch.max(outputs, 1)

                    true = labels.data.view_as(preds)
                    for k in range(len(preds)):
                        confusion[true[k]][preds[k]] += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr = optimizer.param_groups[0]['lr']

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_dec += torch.sum(preds_dec == labels.data)

            if phase == 'train':
                scheduler_ft.step()  # scheduler_ft.step(loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc_dec = running_corrects_dec.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Dec: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc_dec))



            name3 = '/models/checkpoint/FI-loss/CFI1-0701re.pth'
            name4 = '/models/checkpoint/FI-loss/allmodelCFI1-0701re.pth'
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print("The best acc is: ", best_acc)
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, name3)
                torch.save(model, name4)


            if phase == 'val':
                # writer.add_scalar('epoch_valloss', epoch_loss, global_step = epoch)
                # writer.add_scalar('epoch_valacc', epoch_acc, global_step = epoch)
                val_acc_history.append(epoch_acc)
            # else:
            # writer.add_scalar('epoch_trainloss', epoch_loss, global_step = epoch)
            # writer.add_scalar('epoch_traincc', epoch_acc, global_step = epoch)
            # writer.add_scalar('lr_train', lr, global_step = epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling2 = nn.Sequential()
        pooling2.add_module('class_wise', ClassWisePool(num_classes))
        model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)

        input_size = 448

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in
    ['train', 'val']}

class_names = image_datasets['train'].classes
val_number = len(image_datasets['val'])
print(class_names)
print("Total number of test dataset is: ", val_number)

# image_datasets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using GPU ", device)
if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
KL  = nn.KLDivLoss(reduction='batchmean')

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                             is_inception=(model_name == "inception"))

torch.save({
    'model_state_dict': model_ft.state_dict(),
    'optimizer_state_dict': optimizer_ft.state_dict(),
}, model_savepath + 'wscnet.pt')

