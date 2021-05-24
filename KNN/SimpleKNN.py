import numpy as np
import matplotlib.pyplot as plt

def PlotTwoAxisData(Data):
    colors = np.random.rand(len(Data))
    area = (30 * np.random.rand(len(Data))) ** 2
    plt.scatter(trainData_X[:, 0], trainData_X[:, 1], s=area, c=colors, alpha=0.5)
    plt.show()

def KNN(inputData, X_trainData, Y_trainData, para_K):
    distances = np.sqrt(np.power((inputData-X_trainData), 2).sum(axis=1)) # 计算输入数据到各个训练数据的距离
    argSort = distances.argsort() # 获取从小到大排序之后的下标索引
    top_K = Y_trainData[argSort][:para_K]
    return np.bincount(top_K).argmax()

if __name__ == "__main__":
    # 制作训练数据
    trainData_X = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    trainData_Y = np.array([1, 1, 0, 0])

    # PlotTwoAxisData(trainData_X) # 画二维输入数据图

    testData_X = np.array([[0.2, 0.1], [0.22, 0.17]])
    for data in testData_X:
        preLabel = KNN(data, X_trainData=trainData_X, Y_trainData=trainData_Y, para_K=3)
        print(preLabel)

