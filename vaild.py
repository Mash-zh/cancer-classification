import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18  # Example model
import torch.nn as nn
from dataset import CustomDataset

resnet = resnet18()  # Initialize the model architecture
# 2. 修改最后的全连接层为二分类
num_classes = 2  # 二分类
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load('model_weights.pth', weights_only=True))
resnet.eval()  # Set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_dataset = CustomDataset('valid', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0

# No gradient calculation is needed during evaluation
with torch.no_grad():
    for images, labels in valid_loader:
        outputs = resnet(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(predicted)
        # print(labels)
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

