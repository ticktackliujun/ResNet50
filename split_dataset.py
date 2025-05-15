import os
import shutil
import random

# 设置随机种子以保证可重复性
random.seed(42)

# 数据集参数
train_ratio = 0.8  # 80%用于训练，20%用于测试

# 原始数据路径
data_dir = r'D:\pythonPytorch\animaldemo\animals\animals'  # 包含90个子文件夹，每个子文件夹是一个动物类别


# 创建训练集和测试集目录
def create_dataset_structure():
    # 创建目录结构
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/test', exist_ok=True)

    # 为每个类别创建子目录
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # 仅处理文件夹
            os.makedirs(f'dataset/train/{class_name}', exist_ok=True)
            os.makedirs(f'dataset/test/{class_name}', exist_ok=True)


# 划分训练集和测试集
def split_train_test():
    create_dataset_structure()

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # 确保只处理文件夹
        if os.path.isdir(class_path):
            # 获取该类所有图像文件
            images = [f for f in os.listdir(class_path) if
                      f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

            # 检查是否有图片
            if not images:
                print(f"警告: {class_name} 文件夹中没有图片")
                continue

            # 随机打乱
            random.shuffle(images)

            # 计算划分点
            split_point = int(len(images) * train_ratio)

            # 划分训练集和测试集
            train_images = images[:split_point]
            test_images = images[split_point:]

            print(f"{class_name}: 总图片数={len(images)}, 训练集={len(train_images)}, 测试集={len(test_images)}")

            # 复制到相应目录
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join('dataset/train', class_name, img)
                shutil.copy(src, dst)

            for img in test_images:
                src = os.path.join(class_path, img)
                dst = os.path.join('dataset/test', class_name, img)
                shutil.copy(src, dst)

    print("数据集划分完成!")


# 执行划分
split_train_test()