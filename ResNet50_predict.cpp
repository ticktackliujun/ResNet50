#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> loadClassNames(const string& filePath) {
    vector<string> classNames;
    ifstream file(filePath);

    if (!file.is_open()) {
        cerr << "Error: Could not open class names file: " << filePath << endl;
        return classNames;
    }

    string line;
    while (getline(file, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }

    file.close();

    if (classNames.size() != 90) {
        cerr << "Warning: Expected 90 classes, but got " << classNames.size() << endl;
    }

    return classNames;
}

Mat preprocessImage(const string& imagePath) {
    Mat img = imread(imagePath, IMREAD_COLOR); // ȷ����ȡΪ��ɫͼ��
    if (img.empty()) {
        cerr << "Error: Could not read image: " << imagePath << endl;
        return Mat();
    }

    Mat imgResized;
    resize(img, imgResized, Size(224, 224));

    // ���ͨ����
    if (imgResized.channels() != 3) {
        cerr << "Error: Processed image does not have 3 channels" << endl;
        return Mat();
    }


    cvtColor(imgResized, imgResized, COLOR_BGR2RGB);
    
    return imgResized;
}

void predictImage(const string& imagePath, Net& model, const vector<string>& classNames) {
    Mat img = preprocessImage(imagePath);
    if (img.empty()) {
        cerr << "Error: Failed to load or preprocess image." << endl;
        return;
    }

    // ���ͼ��ͨ����
    if (img.channels() != 3) {
        cerr << "Error: Image does not have 3 channels. Actual channels: " << img.channels() << endl;
        return;
    }

    auto start = chrono::high_resolution_clock::now();

    // ����blob�����й�һ����1/255��
    Mat imgRGB;
    cvtColor(img, imgRGB, COLOR_BGR2RGB);  // ��ʽת����ɫ�ռ�
    Mat blob = blobFromImage(imgRGB, 1.0 / 255.0, Size(224, 224), Scalar(0, 0, 0), true, false);

    // ���blobά��
    if (blob.dims != 4 || blob.size[1] != 3) {
        cerr << "Error: Blob dimensions incorrect. Expected [1,3,224,224], got ["
            << blob.size[0] << "," << blob.size[1] << "," << blob.size[2] << "," << blob.size[3] << "]" << endl;
        return;
    }

    // ��׼������ȥ��ֵ�����Ա�׼�
    float* blobData = blob.ptr<float>();
    int totalPixels = 224 * 224;
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float std[] = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < totalPixels; ++i) {
            blobData[c * totalPixels + i] = (blobData[c * totalPixels + i] - mean[c]) / std[c];
        }
    }

    // �������벢ǰ�򴫲�
    model.setInput(blob);
    Mat outputs = model.forward();

    // ������
    if (outputs.empty()) {
        cerr << "Error: Model returned empty output." << endl;
        return;
    }

    // ȷ�����ά��ƥ�������
    if (outputs.total() != classNames.size()) {
        cerr << "Error: Model output dimension (" << outputs.total()
            << ") does not match number of classes (" << classNames.size() << ")" << endl;
        return;
    }

    // ��ȡԤ����
    Mat scores = outputs.reshape(1, 1);  // ȷ����1��N�еľ���
    Mat sortedIndices;
    sortIdx(scores, sortedIndices, SORT_EVERY_ROW + SORT_DESCENDING);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // ��ʾTop-5Ԥ����
    cout << "Top Predictions:" << endl;
    const int numToShow = min(5, static_cast<int>(classNames.size()));

    for (int i = 0; i < numToShow; ++i) {
        int idx = sortedIndices.at<int>(i);
        if (idx >= 0 && idx < classNames.size()) {
            printf("%d. %-20s: %.2f%%\n",
                i + 1,
                classNames[idx].c_str(),
                scores.at<float>(idx) * 100);
        }
    }

    cout << "\nInference time: " << elapsed.count() << " seconds" << endl;
}

int main() {
    const string modelPath = "D:/pythonPytorch/animaldemo/ResNet_best_model_20250514_201625.onnx";
    const string classNamesPath = "D:/pythonPytorch/animaldemo/name_of_the_animals.txt";
    const string imagePath = "D:/pythonPytorch/animaldemo/1.jpg";

    vector<string> classNames = loadClassNames(classNamesPath);
    if (classNames.size() != 90) {
        cerr << "Error: Must have exactly 90 classes" << endl;
        return -1;
    }

    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Error: Failed to load model" << endl;
        return -1;
    }

    // ��ӡģ�����������Ϣ���ڵ���
    vector<String> layerNames = net.getLayerNames();
    cout << "Model layers: " << layerNames.size() << endl;

    predictImage(imagePath, net, classNames);

    return 0;
}