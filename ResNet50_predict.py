import cv2
import numpy as np


def load_class_names(file_path):
    """
    从文件加载类别名称
    :param file_path: 类别名称文件路径
    :return: 类别名称列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        print(f"加载类别名称失败: {str(e)}")
        return None


def load_model(model_path):
    """
    加载ONNX模型
    :param model_path: 模型文件路径
    :return: OpenCV DNN模型
    """
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        if net.empty():
            raise ValueError("模型加载失败，模型为空。")
        return net
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None


def preprocess_image(image_path):
    """
    预处理输入图像，确保尺寸和通道匹配
    :param image_path: 图像路径
    :return: 预处理后的图像
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 确保图像大小为 224x224
        print(f"Original Image Shape: {img.shape}")  # 打印原始图像形状
        img_resized = cv2.resize(img, (224, 224))
        print(f"Resized Image Shape: {img_resized.shape}")  # 打印调整后的图像形状

        return img_resized
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")
        return None


def predict_image(image_path, model, class_names):
    """
    使用OpenCV DNN模型对图像进行预测
    :param image_path: 输入图像路径
    :param model: 加载的OpenCV DNN模型
    :param class_names: 类别名称列表
    :return: 预测结果
    """
    img = preprocess_image(image_path)
    if img is None:
        return None

    try:
        # 创建 blob 对象 - 这里使用与PyTorch相同的预处理方式
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / 255.0,  # 归一化到[0,1]
            size=(224, 224),
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            swapRB=True,  # BGR转RGB
            crop=False
        )

        # 标准化
        blob[0, 0, :, :] = (blob[0, 0, :, :] - 0.485) / 0.229
        blob[0, 1, :, :] = (blob[0, 1, :, :] - 0.456) / 0.224
        blob[0, 2, :, :] = (blob[0, 2, :, :] - 0.406) / 0.225

        model.setInput(blob)
        outputs = model.forward()

        scores = outputs.flatten()
        top_indices = np.argsort(scores)[::-1][:5]
        top_scores = scores[top_indices]

        result = {
            'class_name': class_names[top_indices[0]],
            'confidence': top_scores[0],
            'all_scores': scores
        }
        return result
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None


if __name__ == "__main__":
    ONNX_MODEL_PATH = "ResNet_best_model_20250514_201625.onnx"
    CLASS_NAMES = load_class_names("name_of_the_animals.txt")

    if CLASS_NAMES is None:
        print("加载类别名称失败，程序终止")
    else:
        model = load_model(ONNX_MODEL_PATH)
        if model is None:
            print("加载模型失败，程序终止")
        else:
            result = predict_image("1.jpg", model, CLASS_NAMES)

            if result is None:
                print("预测失败")
            else:
                print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.2%})")
                print("\n所有类别置信度:")
                for i, (name, score) in enumerate(zip(CLASS_NAMES, result['all_scores'])):
                    if i < 5 or score > 0.1:
                        print(f"- {name}: {score:.2%}")