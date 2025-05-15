import torch
import torch.onnx
import onnx
from ResNet import AnimalResNet
# 假设你的模型已经定义并加载了训练好的权重
# 这里假设模型是一个 ResNet 模型
# 假设模型已经保存并且加载
checkpoint = torch.load('ResNet_best_model_20250514_201625.pth')

# 创建并加载模型
# 这里你可以根据实际的模型结构加载
model = AnimalResNet()  # 假设你已经定义了一个ResNet模型
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 切换到评估模式

# 创建一个示例输入张量（假设输入是一个 3x224x224 的图像）
dummy_input = torch.randn(1, 3, 224, 224)

# 定义保存的 ONNX 文件名
onnx_filename = "ResNet_best_model_20250514_201625.onnx"

# 将 PyTorch 模型导出为 ONNX 格式
torch.onnx.export(
    model,               # 需要转换的模型
    dummy_input,         # 模型的输入示例
    onnx_filename,       # 保存的 ONNX 文件名
    export_params=True,  # 是否导出模型参数
    opset_version=12,    # ONNX opset 版本（这里使用12）
    input_names=['input'],  # 输入张量的名字
    output_names=['output'],  # 输出张量的名字
    dynamic_axes={       # 可变维度的支持（如果有）
        'input': {0: 'batch_size'},  # 支持批量大小的动态变化
        'output': {0: 'batch_size'}
    }
)

print(f"Model has been converted to {onnx_filename}")

# 加载并验证转换后的 ONNX 模型
onnx_model = onnx.load(onnx_filename)

# 验证 ONNX 模型的完整性
onnx.checker.check_model(onnx_model)

print("The ONNX model is valid!")

