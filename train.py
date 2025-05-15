import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ResNet import AnimalResNet  # 导入自定义的 ResNet 模型
import os
from datetime import datetime

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 增强的数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random')
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder('dataset/dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder('dataset/dataset/test', transform=test_transform)

# 创建 DataLoader
batch_size = 64  # 适当增大batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 初始化模型
model = AnimalResNet(num_classes=90).to(device)

# 使用混合精度训练和梯度裁剪
scaler = torch.cuda.amp.GradScaler()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # 学习率调度器

# 添加早停机制
early_stopping_patience = 8
best_acc = 0.0
early_stopping_counter = 0

# 训练函数 (使用混合精度训练)
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# 测试函数
def evaluate(model, data_loader, criterion, mode='test'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 混合精度推理
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    eval_loss = running_loss / len(data_loader)
    eval_acc = 100. * correct / total
    return eval_loss, eval_acc

# 主训练循环
def main():
    num_epochs = 100
    best_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 训练日志
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练并获取训练集结果
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        
        # 测试并获取测试集结果
        test_loss, test_acc = evaluate(model, test_loader, criterion, 'test')
        
        # 打印训练和测试结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # 更新学习率
        scheduler.step(test_acc)

        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 早停机制
        if test_acc > best_acc:
            best_acc = test_acc
            early_stopping_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'history': history
            }, f"ResNet_best_model_{timestamp}.pth")
            print("Saved best model!")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

    print(f"\nTraining finished! Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()