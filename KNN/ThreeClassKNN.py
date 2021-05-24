import os
import numpy as np
import pandas as pd

def GetData(fileName = 'AppointmentData.txt'):
    Data_FilePath = os.path.dirname(os.getcwd()) + '/Data/' + fileName
    df = pd.read_csv(Data_FilePath, sep='\t', header=None, names=['x1', 'x2', 'x3', 'y'])
    X_trainData = df[['x1', 'x2', 'x3']].values
    Y_trainData = df['y'].values
    return X_trainData, Y_trainData

def Normal_Data(Data):
    """
    数据归一化：x_norm = (x - x_min)/(x_max - x_min)
    :param Data:
    :return:
    """
    return (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))

def split_Data(DataX, DataY, ratio):
    """
    划分数据集
    :param DataX:
    :param DataY:
    :param ratio: 训练集所占比例
    :return:
    """
    trainDataLen = int(len(DataX) * ratio)
    X_trainData = DataX[:trainDataLen]
    Y_trainData = DataY[:trainDataLen]
    X_testData = DataX[trainDataLen:]
    Y_testData = DataY[trainDataLen:]
    return X_trainData, Y_trainData, X_testData, Y_testData

def KNN(inputData, X_trainData, Y_trainData, para_K):
    distances = np.sqrt(np.power((inputData-X_trainData), 2).sum(axis=1)) # 计算输入数据到各个训练数据的距离
    argSort = distances.argsort() # 获取从小到大排序之后的下标索引
    top_K = Y_trainData[argSort][:para_K]
    return np.bincount(top_K).argmax()

if __name__ == "__main__":
    Data_X, Data_Y = GetData(fileName = 'AppointmentData.txt')
    Data_X = Normal_Data(Data_X)
    trainData_X, trainData_Y, testData_X, testData_Y = split_Data(Data_X, Data_Y, ratio=0.9)

    CountCorrect = 0
    for x, y in zip(testData_X,testData_Y):
        pre_y = KNN(x, trainData_X, trainData_Y, para_K=5)
        if pre_y == y: CountCorrect += 1
    print("Correct Accuracy is : {}".format(CountCorrect / len(testData_X)))
