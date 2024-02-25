import csv
import os
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
normalize=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
test_transforms=transforms.Compose([
    transforms.Resize((228,228)),
    transforms.ToTensor(),
    normalize
])
train_dataset=ImageFolder("val",transform=test_transforms)
a=train_dataset[0]
my_resnet=torchvision.models.resnet50(pretrained=True)
my_resnet.fc=torch.nn.Linear(2048,2)
model=torch.load("resnet25.pth",map_location=torch.device('cpu'))
model.eval()
# 假设你有一些数据，比如一个列表的列表，每个列表代表一行数据
# 指定要写入的 CSV 文件的路径
csv_file_path = 'vali.csv'
# 使用 'w' 模式打开文件，创建 CSV writer 对象
data=['ID','TARGET']
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 使用 writerows() 方法将数据写入 CSV 文件
    writer.writerow(data)
num=1
step=1
root_dir="val/test/"
writer2 = SummaryWriter(log_dir='logs')
with torch.no_grad():
    for i in range(9375):
        img_num=str(i+1)+".jpg"
        img_path=os.path.join(root_dir,img_num)
        img=Image.open(img_path)
        img=test_transforms(img)
        writer2.add_image("test_data", img, step)
        step+=1
        writer2.close()
        img = img.unsqueeze(0)
        output = model(img)
        output=output.sigmoid()
        output=output.tolist()
        pro=output[0][1]
        con=[num,pro]
        num+=1
        with open(csv_file_path, 'a', newline='\n') as csv_file:
            writer = csv.writer(csv_file)
            # 使用'' writerows() 方法将数据写入 CSV 文件
            writer.writerow(con)


