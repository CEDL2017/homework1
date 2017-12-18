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
save_path_and_model_name = "Two_stream_ResNet-152_FE_batch64_dropout03"
batch_size = 64
num_epochs = 25
threshold = 999
dropout = 0.3
step_size=7

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


def train_model(model0, model1, model2, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()

    best_model_wts = model2.state_dict()
    best_acc = 0.0
    torch.save(model2.state_dict(), "save/{}/{}_model.pkl".format(save_path_and_model_name, save_path_and_model_name))
    print("save")
    epoch_list = []
    epoch_loss_list = []
    epoch_acc_list = []
    early_stop_count = 0
    global threshold


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_list.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model0.train(False)  # Set model to training mode
                model1.train(False)  # Set model to training mode
                model2.train(True)
            else:
                model0.train(False)  # Set model to evaluate mode
                model1.train(False)  # Set model to evaluate mode
                model2.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

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
                optimizer.zero_grad()

                # forward
                _, outputs0 = model0(inputs0)
                _, outputs1 = model1(inputs1)
                outputs0 = outputs0.data
                outputs1 = outputs1.data
                two_stream_inputs = Variable(torch.cat((outputs0, outputs1), 1))
                two_stream_outputs = model2(two_stream_inputs)

                _, preds = torch.max(two_stream_outputs.data, 1)
                loss = criterion(two_stream_outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_list.append(epoch_loss)
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_acc_list.append(epoch_acc)

            print('{} Loss model: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            f.write('{} Loss model: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model2.state_dict()
                torch.save(model2.state_dict(), "save/{}/{}_model.pkl".format(save_path_and_model_name, save_path_and_model_name))
                print("save")
                f.write("save\n")
                early_stop_count = 0
            elif phase == 'test' and epoch_acc < best_acc:
                early_stop_count += 1

        if early_stop_count >= threshold:
            break

        print()

    np.save("save/{}/{}_epoch_list.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_list))
    np.save("save/{}/{}_epoch_loss_list.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_loss_list))
    np.save("save/{}/{}_epoch_acc_list.npy".format(save_path_and_model_name, save_path_and_model_name), np.array(epoch_acc_list))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))
    f.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model2.load_state_dict(best_model_wts)
    return model0, model1, model2


# def visualize_model(model0, model1, model2, num_images=12):
#     images_so_far = 0
#     fig = plt.figure()
#
#     for i, data in enumerate(dataloders['test']):
#         inputs0, inputs1, labels = data
#         if use_gpu:
#             inputs0, inputs1, labels = Variable(inputs0.cuda()), Variable(inputs1.cuda()), Variable(labels.cuda())
#         else:
#             inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels)
#
#         # forward
#         _, outputs0 = model0(inputs0)
#         _, outputs1 = model1(inputs1)
#         outputs0 = outputs0.data
#         outputs1 = outputs1.data
#         two_stream_inputs = Variable(torch.cat((outputs0, outputs1), 1))
#         two_stream_outputs = model2(two_stream_inputs)
#
#         _, preds = torch.max(two_stream_outputs.data, 1)
#
#         for j in range(inputs0.size()[0]):
#             images_so_far += 1
#             ax = plt.subplot(num_images//2, 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predicted: {}'.format(preds[j]))
#             imshow(inputs0.cpu().data[j], discribe="{}_visualize_model_hand".format(save_path_and_model_name))
#
#             images_so_far += 1
#             ax = plt.subplot(num_images//2, 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predicted: {}'.format(preds[j]))
#             imshow(inputs1.cpu().data[j], discribe="{}_visualize_model_head".format(save_path_and_model_name))
#
#             if images_so_far == num_images:
#                 return


######################################################################
from pretrained_models import pretrainedmodels

model_name = 'fbresnet152' #fbresnet152
model_conv0 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

for param in model_conv0.parameters():
    param.requires_grad = False


if use_gpu:
    model_conv0 = model_conv0.cuda()
#####################################################################
#####################################################################
model_name = 'fbresnet152' #fbresnet152
model_conv1 = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
for param in model_conv1.parameters():
    param.requires_grad = False


if use_gpu:
    model_conv1 = model_conv1.cuda()
######################################################################
features0_dim = model_conv0.fc.in_features
features1_dim = model_conv1.fc.in_features

two_stream_ResNet152 = torch.nn.Sequential(
    torch.nn.Linear(features0_dim + features1_dim, features0_dim + features1_dim),
    torch.nn.Dropout(dropout),  # drop 30% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(features0_dim + features1_dim, features0_dim),
    torch.nn.Dropout(dropout),  # drop 30% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(features0_dim, 24),
)

if use_gpu:
    two_stream_ResNet152 = two_stream_ResNet152.cuda()

######################################################################
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer = optim.Adam(two_stream_ResNet152.parameters(), lr=0.001)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)


######################################################################
# Train and evaluate
model0, model1, two_stream_model = train_model(model_conv0, model_conv1, two_stream_ResNet152, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=num_epochs)

######################################################################
# visualize_model(model0, model1, two_stream_model)

f.close()


