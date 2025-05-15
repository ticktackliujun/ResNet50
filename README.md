# ResNet50
这个项目的目标是使用Python和PyTorch框架微调ResNet的预训练模型，之后导出为ONNX格式，并在C++中使用OpenCV的DNN模块进行推理
以下是按照项目流程顺序的具体文件说明：
Python：
1.split_dataset.py文件用于划分原始数据集,数据标签来源于your_animals_dataset.txt(我的数据集为90个动物类别),请根据自己另行下载,该数据集来源于Kaggle平台
2.ResNet.py为接下来的训练中需要引入的模型文件,需要依赖PyTorch框架
3.train.py为训练文件,导入了上面ResNet.py中自定义的ResNet模型,模型导出格式为.pth格式(该格式一般为Pytorch训练框架的文件格式)
4.change_toonnx.py用于将前面训练完成的.pth模型文件保存为.onnx(推理格式)
5.ResNet50_predict.py导入.onnx文件和your_animals_dataset.txt,使用Python环境第三方库opencv的dnn模块进行推理

C++：ResNet50_predict.cpp为C++环境下的推理代码,同样需要导入.onnx文件和your_animals_dataset.txt,使用C++环境第三方库opencv的dnn模块进行推理

注：1.数据集并未提供,请确定好自己的任务使用相应的数据集和对应数据标签文件
    2.注意环境为Python的Pytorch(GPU)版本和opencv的dnn模块,C++的opencv的dnn模块
