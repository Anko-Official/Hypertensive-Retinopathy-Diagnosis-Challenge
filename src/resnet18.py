import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

# 检查是否有可用的GPU，如果有，则使用第一个可用的GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子，确保结果可重复性
torch.manual_seed(0)

# 数据预处理
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机裁剪和缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色扰动
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载数据集
# 这里假设你已经有一个 PyTorch 数据集类 EyeDataset，你可以根据自己的数据集进行替换
data_dir = 'C:/Users/Lynn/Desktop/2-Hypertensive Retinopathy Classification/1-Images/dataset'
dataset = datasets.ImageFolder(data_dir, transform=transform)

# 创建数据加载器
train_size = int(1 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 加载 ResNet-18 模型
model = torchvision.models.resnet18(pretrained=True)

# 更改最后的全连接层以适应新的任务
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 假设 num_classes 是你的输出类别数量

model.to(device)

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练模型
print('Start training!')
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # model.eval()
    # running_corrects = 0
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         outputs = model(inputs)
    #         preds = torch.argmax(outputs, 1)
    #         running_corrects += torch.sum(preds == labels.data)
    #         class_0_preds = preds[labels == 0]
    #         class_1_preds = preds[labels == 1]
    #
    # epoch_acc = float(running_corrects) / len(val_dataset)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    if (epoch + 1) % 10 == 0:
        torch.save(model, f'./model/resnet18_{epoch + 1}.pth')