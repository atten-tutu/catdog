import os

import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image, ImageChops
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

device = torch.device("cuda")

#模型搭建
my_resnet=torchvision.models.resnet50(pretrained=True)
# for param in my_resnet.parameters():
#     param.requires_grad = False
my_resnet.fc=torch.nn.Linear(2048,2)

# my_resnet.sig=torch.nn.Sigmoid()
# # print(my_resnet)
# my_resnet=my_resnet.to(device)

# class my_alexnet(nn.Module):
#     def __init__(self):
#         super(my_alexnet, self).__init__()
#         self.conv1 =Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
#         self.maxpool=MaxPool2d(kernel_size=3, stride=2)
#         self.conv2 =Conv2d(in_channels=96, out_channels=256, kernel_size=5,padding=2)
#         self.conv3 =Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
#         self.conv4 = Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
#         self.conv5 = Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
#         self.linear1 = Linear(1,4096)
#         self.linear2 = Linear(4096, 2048)
#         self.linear3 = Linear(2048, 1000)
#         self.linear4 = Linear(1000,2)
#     def forward(self, x):
#         x=F.relu(self.maxpool(self.conv1(x)))
#         x=F.relu(self.maxpool(self.conv2(x)))
#         x=F.relu(self.conv3(x))
#         x=F.relu(self.conv4(x))
#         x=F.relu(self.conv5(x))
#         x=self.maxpool(x)
#         x=self.linear1(x)
#         x=self.linear2(x)
#         x=self.linear3(x)
#         x=self.linear4(x)
#         return x
#数据增强
normalize=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_transforms=transforms.Compose([
    # transforms.CenterCrop(224),
    # transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),#裁剪成任意大小和纵横比
    # transforms.RandomVerticalFlip(p=0.5),#垂直翻转
    # transforms.RandomHorizontalFlip(p=0.5),#水平翻转
    # transforms.RandomRotation(degrees=45), #随机旋转
    # transforms.Pad(padding=(10,20,30,40),fill=(255,255,255)),
    # transforms.RandomPerspective(distortion_scale=0.5,p=1.0),
    # transforms.RandomAffine(degrees=45),

    transforms.RandomRotation(45,expand=True),
    transforms.Resize((228,228)),
    # # # # # transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),#随机改变亮度饱和度对比度
    transforms.RandomInvert(p=1),
    transforms.ToTensor(),
    normalize,
])

# data_dir="D:\learn_pytorch\classification"
#
# train_datasets=datasets.ImageFolder(root=os.path.join(data_dir,"ttrain"),transform=train_transforms)
# test_datasets=datasets.ImageFolder(root=os.path.join(data_dir,"ttest"),transform=test_transforms)


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = str(os.path.join(self.root_dir, self.label_dir))
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.img_path[idx].split(".")[0]
        img=train_transforms(img)
        if label == "dog":
            # label = torch.tensor(1)
            label=1
        else:
            # label = torch.tensor(0)
            label=0
        return img, label



root_dir = "classification"
train_label_dir = "train"
# test_label_dir = "train"
train_data = MyDataset(root_dir, train_label_dir)

train_loader = DataLoader(dataset=train_data, shuffle=True,batch_size=32, num_workers=0, drop_last=False)
# example_classes=train_datasets.classes
# index_classes=train_datasets.class_to_idx
# train_dataset=ImageFolder("classification",transform=train_transforms)
# test_dataset=ImageFolder("train_cla",transform=test_transforms)
# train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
# test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
# #length
train_data_size=len(train_data)
# test_data_size=len(test_data)

# img = Image.open("classification/train/cat.12.jpg")
# writer = SummaryWriter("logs")
# trans=train_transforms(img)
# writer.add_image("improve", trans)
# writer.close()
# model=my_alexnet()
# criterion=nn.CrossEntropyLoss()
# optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
# lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
# def
#创建网络模型

#损失函数优化器
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)
learing_rate=0.001
optimizer=torch.optim.SGD(my_resnet.parameters(),lr=learing_rate)
#训练网络参数 ny
total_train_step=0
total_test_step=0
epoh=10
# writer=SummaryWriter("logs")
# step=0
for i in range(epoh):
    print("第{}轮训练开始".format(i+1))
    total_test_loss = 0
    total_accuracy = 0
    for data in train_loader:
        imgs,targets=data
        # writer.add_images("test",imgs,step)
        # writer.close()
        # step+=1
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs=my_resnet(imgs)
        outputs=outputs.sigmoid()
        loss=loss_fn(outputs,targets)
        total_test_loss += loss.item()
        optimizer.zero_grad()
        loss.requires_grad_(True)  # 加入
        loss.backward()
        optimizer.step()
        total_train_step+=1

        if total_train_step %100==0:
            print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
            # writer.add_scalar("train_loss",loss.item(),total_train_step)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy +=accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/train_data_size))
    total_test_step+=1
    torch.save(my_resnet,"resnet{}.pth".format(i))
    print("模型保存")

# #模型验证
# my_resnet.eval()
# root_dir = "classification"
# test_label_dir = "test"
# test_data = MyDataset2(root_dir, train_label_dir)
# test_loader = DataLoader(dataset=train_data, shuffle=True, num_workers=0, drop_last=False)
# test_data_size=len(test_data)
# model = torch.load("")
# for data in test_loader:
#     images,targets=data
#     output=model(images)
#     print(output.argmax(1))
#     x=output.argmax(1)
#     if(x==1):
#         print(output[1])
#     else:
#         print(output[0])