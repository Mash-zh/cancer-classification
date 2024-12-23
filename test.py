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

def test(net_name, model_path, data_path, batch_size):
    model = CustomNet(net_name)
    model = model.__net__()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    # 3. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # ResNet输入固定224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载数据集
    test_dataset = CustomDataset(data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    with torch.no_grad():
        for images , labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            total_correct += sum(outputs == labels)
    accuracy = total_correct.item() / len(test_dataset)
    print(accuracy)

if __name__ == '__main__':
    # epochs = 100
    # data_path = 'train_all'
    # # 'resnet18' 'resnet34' 'vgg16' 'efficientnet_v2_m' 'inception3'
    # net_list = ['resnet18', 'resnet34', 'vgg16', 'efficientnet_v2_m', 'inception3']
    # for net_name in net_list:
    #     loss_csv = net_name+'_loss.csv'
    #     train(epochs, net_name, loss_csv, data_path)
    model_path = './resnet18_model_weights_0.15847289782047272.pth'
    data_path = 'test'
    net_name = 'resnet18'
    test(net_name=net_name, model_path=model_path, data_path=data_path, batch_size=32)