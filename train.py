import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataset import CustomDataset
import pandas as pd
from net import CustomNet
import glob

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 训练循环
def train(epochs, net_name, loss_csv, data_path, save_csv, save_weight, model_path):
    if net_name == 'nas':
        net = torch.load(model_path)
    else:
        net = CustomNet(net_name)
        net = net.__net__()

    # 检查是否有GPU并加载模型到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 3. 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(30),  # 随机旋转
        # transforms.RandomResizedCrop(32),  # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载数据集（示例使用ImageFolder）
    train_dataset = CustomDataset(data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        dataframe = pd.DataFrame({'epoch': [epoch+1], 'loss': [running_loss/len(train_loader)]})
        if os.path.exists(loss_csv):
            if save_csv:
                dataframe.to_csv(loss_csv, mode='a', header=False, index=False)
        else:
            min_loss = running_loss/len(train_loader)
            if save_csv:
                dataframe.to_csv(loss_csv, index=False)
            if save_weight:
                model_name = net_name+'_model_weights_'+str(min_loss)+'.pth'
                if net_name == 'nas':
                    torch.save(net, model_name)
                else:
                    torch.save(net.state_dict(), model_name)
        if min_loss > running_loss/len(train_loader):
            min_loss = running_loss/len(train_loader)
            files = glob.glob(net_name+'_model_weights_*')
            for file in files:
                os.remove(file)
            model_name = net_name+'_model_weights_'+str(min_loss)+'.pth'
            if net_name == 'nas':
                torch.save(net, model_name)
            else:
                torch.save(net.state_dict(), model_name)
        torch.cuda.empty_cache()
    print(net_name+"训练完成！")

if __name__ == '__main__':
    epochs = 20
    data_path = 'data/cancer/train'
    # 'resnet18' 'resnet34' 'vgg16' 'efficientnet_v2_m' 'inception3'
    save_csv = True
    save_weight = True
    # net_list = ['resnet18', 'resnet34', 'vgg16', 'efficientnet_v2_m', 'inception3', 'nas']
    net_list = ['nas']
    model_path = 'searchs/test/best.pth.tar'
    for net_name in net_list:
        loss_csv = net_name+'_loss.csv'
        train(epochs, net_name, loss_csv, data_path, save_csv, save_weight, model_path)