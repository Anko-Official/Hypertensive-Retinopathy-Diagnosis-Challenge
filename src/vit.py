import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from transformers import ViTForImageClassification

# 检查是否有可用的GPU，如果有，则使用第一个可用的GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子，确保结果可重复性
torch.manual_seed(0)

# 数据预处理和增强
data_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机裁剪和缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色扰动
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载数据集
data_dir = 'C:/Users/Lynn/Desktop/2-Hypertensive Retinopathy Classification/1-Images/dataset'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# train_target_count = 292
# train_indices = []
# for class_idx in dataset.classes:
#     class_indices = np.where(np.array(dataset.targets) == int(class_idx))[0]
#     class_indices = np.random.choice(class_indices, size=train_target_count, replace=False)
#     train_indices.extend(class_indices)
# val_indices = list(set(range(len(dataset))) - set(train_indices))
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
# train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
# val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)


# 动态划分数据集为训练集和验证集，这里假设训练集占80%，验证集占20%
train_size = int(1 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


model = ViTForImageClassification.from_pretrained("config/vit", local_files_only=True)
# print(model)
# 冻结所有预训练的层参数
# for param in model.parameters():
#     param.requires_grad = False
# 修改最后一层，使其适应你的分类任务
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(dataset.classes)).to(device)
model.classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=2),
    nn.Softmax(dim=1)
)
model.to(device)

# total = sum([param.nelement() for param in model.parameters()])
# print(total)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
if __name__ == "__main__":
    print('Start training!')
    # print(f'Class 0: {len([sample for sample in val_dataset if sample[1] == 0])}')
    # print(f'Class 1: {len([sample for sample in val_dataset if sample[1] == 1])}')

    num_epochs = 50
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # 测试阶段
        # model.eval()
        # running_corrects = 0
        # class_0_corrects = 0
        # class_1_corrects = 0
        # best_acc = 10000
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #
        #         outputs = model(inputs).logits
        #         preds = torch.argmax(outputs, 1)
        #         running_corrects += torch.sum(preds == labels.data)
        #         class_0_preds = preds[labels == 0]
        #         class_1_preds = preds[labels == 1]
        #         class_0_corrects += torch.sum(class_0_preds == 0)
        #         class_1_corrects += torch.sum(class_1_preds == 1)
        #
        # epoch_acc = running_corrects.double() / len(val_dataset)
        # class_0_acc = class_0_corrects.double() / len([sample for sample in val_dataset if sample[1] == 0])
        # class_1_acc = class_1_corrects.double() / len([sample for sample in val_dataset if sample[1] == 1])
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}, Class 0: {class_0_acc:.4f}, Class 1: {class_1_acc:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model, f'./model/vit_{epoch+1}.pth')
