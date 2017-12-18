import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
FOLDER_DATASET = "data/frames"
from glob import glob

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import time
import os
import pickle

#*****************************************************************
save_path_and_model_name = "ResNet-152_FE_batch256"
batch_size = 256
num_epochs = 25
threshold = 999

if not os.path.exists("save/{}".format(save_path_and_model_name)):
    os.makedirs("save/{}".format(save_path_and_model_name))

#*****************************************************************

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

np.random.seed(1)


train_X_list = []
train_y_list = []
with open("save/datasets/train_X_list.pickle", "rb") as f:
    train_X_list = pickle.load(f)
with open("save/datasets/train_y_list.pickle", "rb") as f:
    train_y_list = pickle.load(f)

test_X_list = []
test_y_list = []
with open("save/datasets/test_X_list.pickle", "rb") as f:
    test_X_list = pickle.load(f)
with open("save/datasets/test_y_list.pickle", "rb") as f:
    test_y_list = pickle.load(f)

print(len(train_X_list), len(train_y_list))
print(len(test_X_list), len(test_y_list))


class Two_stream_data_train(Dataset):
    __Xs0 = []
    __Xs1 = []
    __ys = []

    def __init__(self, Xs, ys, transform=None):
        self.transform = transform
        for X, y in zip(Xs, ys):
            # Image path
            self.__Xs0.append(X[0])
            self.__Xs1.append(X[1])
            self.__ys.append(y)

    def __getitem__(self, index):
        img0 = Image.open(self.__Xs0[index])
        # img0 = img0.convert('RGB')
        if self.transform is not None:
            img0 = self.transform(img0)

        img1 = Image.open(self.__Xs1[index])
        # img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)

        # Convert images and label to torch tensors
        # img = torch.from_numpy(np.asarray(img))
        # label = torch.from_numpy(np.asarray(self.__ys[index]).reshape(1))
        label = self.__ys[index]
        return img0, img1, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__Xs0)


################################################################################################
class Two_stream_data_test(Dataset):
    __Xs0 = []
    __Xs1 = []
    __ys = []

    def __init__(self, Xs, ys, transform=None):
        self.transform = transform
        for X, y in zip(Xs, ys):
            # Image path
            self.__Xs0.append(X[0])
            self.__Xs1.append(X[1])
            self.__ys.append(y)

    def __getitem__(self, index):
        img0 = Image.open(self.__Xs0[index])
        # img0 = img0.convert('RGB')
        if self.transform is not None:
            img0 = self.transform(img0)

        img1 = Image.open(self.__Xs1[index])
        # img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)

        # Convert images and label to torch tensors
        # img = torch.from_numpy(np.asarray(img))
        # label = torch.from_numpy(np.asarray(self.__ys[index]).reshape(1))
        label = self.__ys[index]
        return img0, img1, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__Xs0)
################################################################################################
transform_train = transforms.Compose([
        transforms.Scale(256), transforms.RandomCrop((224, 398)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose([
        transforms.Scale(256), transforms.CenterCrop((224, 398)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
################################################################################################


dset_train = Two_stream_data_train(train_X_list, train_y_list, transform = transform_train)
dset_test = Two_stream_data_test(test_X_list, test_y_list, transform = transform_test)

train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=6) # <<
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=True, num_workers=6) # <<

def imshow(inp, title=None, discribe="xxx"):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # plt.imshow(inp)
    plt.imsave("save/{}/{}.png".format(save_path_and_model_name, discribe), inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.1)

# img0, img1, labels = next(iter(train_loader))
# print('Batch shape0:',img0.numpy().shape)
# print('Batch shape1:',img1.numpy().shape)
# imshow(img0[0,:,:,:], discribe="im00")
# imshow(img1[0,:,:,:], discribe="im10")
# imshow(img0[-1,:,:,:], discribe="im0-1")
# imshow(img1[-1,:,:,:], discribe="im1-1")
# # plt.ioff()
# print(labels)
#
# img0, img1, labels = next(iter(test_loader))
# print('Batch shape:',img0.numpy().shape)
# print('Batch shape1:',img1.numpy().shape)
# imshow(img0[0,:,:,:], discribe="im00")
# imshow(img1[0,:,:,:], discribe="im10")
# imshow(img0[-1,:,:,:], discribe="im0-1")
# imshow(img1[-1,:,:,:], discribe="im1-1")
# # plt.ioff()
# print(labels)


################################################################################################
################################################################################################
################################################################################################


# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Scale(232), transforms.RandomCrop((224, 398)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
#     'test': transforms.Compose([
#         transforms.Scale(232), transforms.CenterCrop((224, 398)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
# }

# data_dir = 'data/trans_data'

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'test']}

dataloders = {'train': train_loader,
              'test': test_loader }

dataset_sizes = {'train': len(dset_train),
                 'test': len(dset_test)}

f = open("save/{}/{}.txt".format(save_path_and_model_name, save_path_and_model_name), "w")

print("train size: {}, test size: {}".format(dataset_sizes["train"], dataset_sizes["test"]))
f.write("train size: {}, test size: {}\n".format(dataset_sizes["train"], dataset_sizes["test"]))
# class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# Get a batch of training data
inputs0, inputs1, classes = next(iter(dataloders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs0)
imshow(out, title=[x for x in classes], discribe="{}_train_batch_hand".format(save_path_and_model_name))
out = torchvision.utils.make_grid(inputs1)
imshow(out, title=[x for x in classes], discribe="{}_train_batch_head".format(save_path_and_model_name))

# Get a batch of training data
inputs0, inputs1, classes = next(iter(dataloders['test']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs0)
imshow(out, title=[x for x in classes], discribe="{}_test_batch_hand".format(save_path_and_model_name))
out = torchvision.utils.make_grid(inputs1)
imshow(out, title=[x for x in classes], discribe="{}_test_batch_head".format(save_path_and_model_name))

# model_convc0, model_convc1 = train_model(model_conv0, model_conv1, criterion, optimizer_conv0, optimizer_conv1,
#                          exp_lr_scheduler0, exp_lr_scheduler1, num_epochs=25)
def train_model(model0, model1, criterion, optimizer0, optimizer1, scheduler0, scheduler1, num_epochs=num_epochs):
    since = time.time()

    best_model0_wts = model0.state_dict()
    best_model1_wts = model1.state_dict()
    best_acc0 = 0.0
    best_acc1 = 0.0
    torch.save(model0.state_dict(), "save/{}/{}_model0.pkl".format(save_path_and_model_name, save_path_and_model_name))
    torch.save(model1.state_dict(), "save/{}/{}_model1.pkl".format(save_path_and_model_name, save_path_and_model_name))
    print("save")
    epoch_list = []
    epoch_loss_list0 = []
    epoch_loss_list1 = []
    epoch_acc_list0 = []
    epoch_acc_list1 = []
    early_stop_count = 0
    global threshold


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_list.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler0.step()
                scheduler1.step()
                model0.train(True)  # Set model to training mode
                model1.train(True)  # Set model to training mode
            else:
                model0.train(False)  # Set model to evaluate mode
                model1.train(False)  # Set model to evaluate mode

            running_loss0 = 0.0
            running_loss1 = 0.0
            running_corrects0 = 0
            running_corrects1 = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs0, inputs1, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs0 = Variable(inputs0.cuda())
                    inputs1 = Variable(inputs1.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels)

                # zero the parameter gradients
                optimizer0.zero_grad()
                optimizer1.zero_grad()

                # forward
                outputs0 = model0(inputs0)
                outputs1 = model1(inputs1)
                _, preds0 = torch.max(outputs0.data, 1)
                _, preds1 = torch.max(outputs1.data, 1)
                loss0 = criterion(outputs0, labels)
                loss1 = criterion(outputs1, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss0.backward()
                    loss1.backward()
                    optimizer0.step()
                    optimizer1.step()

                # statistics
                running_loss0 += loss0.data[0]
                running_loss1 += loss1.data[0]
                running_corrects0 += torch.sum(preds0 == labels.data)
                running_corrects1 += torch.sum(preds1 == labels.data)

            epoch_loss0 = running_loss0 / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss_list0.append(epoch_loss0)
            epoch_loss_list1.append(epoch_loss1)
            epoch_acc0 = running_corrects0 / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]
            epoch_acc_list0.append(epoch_acc0)
            epoch_acc_list1.append(epoch_acc1)

            print('{} Loss model0: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss0, epoch_acc0))
            print('{} Loss model1: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss1, epoch_acc1))
            f.write('{} Loss model0: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss0, epoch_acc0))
            f.write('{} Loss model1: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss1, epoch_acc1))

            # deep copy the model
            if phase == 'test' and epoch_acc0 > best_acc0:
                best_acc0 = epoch_acc0
                best_model0_wts = model0.state_dict()
                torch.save(model0.state_dict(), "save/{}/{}_model0.pkl".format(save_path_and_model_name, save_path_and_model_name))
                print("save")
                f.write("save\n")
                early_stop_count = 0
            elif phase == 'test' and epoch_acc0 < best_acc0:
                early_stop_count += 1

            if phase == 'test' and epoch_acc1 > best_acc1:
                best_acc1 = epoch_acc1
                best_model1_wts = model1.state_dict()
                torch.save(model1.state_dict(), "save/{}/{}_model1.pkl".format(save_path_and_model_name, save_path_and_model_name))
                print("save")
                f.write("save\n")
                early_stop_count = 0
            elif phase == 'test' and epoch_acc1 < best_acc1:
                early_stop_count += 1

        if early_stop_count >= threshold:
            break

        print()

    np.save("save/{}/{}_epoch_list.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_list))
    np.save("save/{}/{}_epoch_loss_list0.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_loss_list0))
    np.save("save/{}/{}_epoch_loss_list1.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_loss_list1))
    np.save("save/{}/{}_epoch_acc_list0.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_acc_list0))
    np.save("save/{}/{}_epoch_acc_list1.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_acc_list1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc0: {:4f}'.format(best_acc0))
    print('Best val Acc1: {:4f}'.format(best_acc1))
    f.write('Best val Acc0: {:4f}\n'.format(best_acc0))
    f.write('Best val Acc1: {:4f}\n'.format(best_acc1))

    # load best model weights
    model0.load_state_dict(best_model0_wts)
    model1.load_state_dict(best_model1_wts)
    return model0, model1


def visualize_model(model0, model1, num_images=12):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloders['test']):
        inputs0, inputs1, labels = data
        if use_gpu:
            inputs0, inputs1, labels = Variable(inputs0.cuda()), Variable(inputs1.cuda()), Variable(labels.cuda())
        else:
            inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels)

        # forward
        outputs0 = model0(inputs0)
        outputs1 = model1(inputs1)
        _, preds0 = torch.max(outputs0.data, 1)
        _, preds1 = torch.max(outputs1.data, 1)

        for j in range(inputs0.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted0: {}'.format(preds0[j]))
            imshow(inputs0.cpu().data[j], discribe="{}_visualize_model0_hand".format(save_path_and_model_name))

            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted1: {}'.format(preds1[j]))
            imshow(inputs1.cpu().data[j], discribe="{}_visualize_model1_head".format(save_path_and_model_name))

            if images_so_far == num_images:
                return


######################################################################
model_conv0 = torchvision.models.resnet152(pretrained=True)
model_conv0.avgpool = nn.AdaptiveAvgPool2d(1)
for param in model_conv0.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs0 = model_conv0.fc.in_features
model_conv0.fc = nn.Linear(num_ftrs0, 24)

if use_gpu:
    model_conv0 = model_conv0.cuda()
#####################################################################
#####################################################################
model_conv1 = torchvision.models.resnet152(pretrained=True)
model_conv1.avgpool = nn.AdaptiveAvgPool2d(1)
for param in model_conv1.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs1 = model_conv1.fc.in_features
model_conv1.fc = nn.Linear(num_ftrs1, 24)

if use_gpu:
    model_conv1 = model_conv1.cuda()
######################################################################
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv0 = optim.Adam(model_conv0.fc.parameters(), lr=0.001)
optimizer_conv1 = optim.Adam(model_conv1.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler0 = lr_scheduler.StepLR(optimizer_conv0, step_size=7, gamma=0.1)
exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer_conv1, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
model_convc0, model_convc1 = train_model(model_conv0, model_conv1, criterion, optimizer_conv0, optimizer_conv1,
                         exp_lr_scheduler0, exp_lr_scheduler1, num_epochs=num_epochs)

######################################################################
visualize_model(model_convc0, model_convc1)

f.close()


