import os
import numpy as np
import pandas as pd

def GetData(fileName = 'WatermelonOriginalWithContinuous.txt'):
    Data_FilePath = os.path.dirname(os.getcwd()) + '/Data/' + fileName
    df = pd.read_csv(Data_FilePath, sep=',', header=None, names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8','y'])
    X_trainData = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8',]].values
    Y_trainData = df['y'].values
    return X_trainData, Y_trainData

def split_Data(DataX, DataY, ratio):
    """
    划分数据集
    :param DataX:
    :param DataY:
    :param ratio: 训练集所占比例
    :return:
    """
    trainDataLen = int(len(DataX) * ratio)
    X_trainData = DataX[:]
    Y_trainData = DataY[:]
    X_testData = DataX[trainDataLen:]
    Y_testData = DataY[trainDataLen:]
    return X_trainData, Y_trainData, X_testData, Y_testData

def ComputePostProbability(DataX, DataY, Col, Value, Y):
    """
    给定Y下计算 P(x_i | Y)
    :param DataX:
    :param DataY:
    :param Col:
    :param Value:
    :param Y:
    :return:
    """
    SubData = DataX[DataY==Y] #首先根据Y分割出X的子集
    # 分子是在SubData中, col=value的数量,+1是做拉普拉斯平滑
    CountX_i = len(SubData[SubData[:,Col] == Value]) + 1
    # 分母是SubData的数量, 加col列的取值数量做拉普拉斯平滑
    Count_X = len(SubData) + len(np.unique(SubData[:, Col]))
    return CountX_i / Count_X

def ComputePostProbabilityContinuous(DataX, DataY, Col, Value, Y):
    """
    给定Y下计算 P(x_i | Y) -> 连续值情况
    :param DataX:
    :param DataY:
    :param Col:
    :param Value:
    :param Y:
    :return:
    """
    sqrt_2_pi = (2 * np.pi)**0.5
    SubData = DataX[DataY == Y]  # 首先根据Y分割出X的子集
    mu = SubData[:, Col].mean()
    sigma = SubData[:, Col].std()
    # 计算第一部分
    p = 1 / (sqrt_2_pi * sigma)
    # 计算第二部分
    fenzi = (Value - mu) ** 2
    fenmu = sigma ** 2 * 2
    p *= np.exp(-fenzi / fenmu)
    return p

def predict(InputX, DataX, DataY):
    """
    依次计算，给定Y 的 P(x_i | Y)
    :param InputX:
    :param DataX:
    :param DataY:
    :return:
    """
    M = DataX.shape[1]
    # 结果是两个概率,因为是对数概率,所以初始化为0, 如果不是对数就需要初始化为1
    ps = np.zeros(2)
    for label in range(2):
        for col in range(M):
            if col == 6 or col == 7: # 6和7是连续值,其他的是离散值
                p = ComputePostProbabilityContinuous(DataX=DataX, DataY=DataY, Col=col, Value=InputX[col], Y=label)
            else:
                p = ComputePostProbability(DataX=DataX, DataY=DataY, Col=col, Value=InputX[col], Y=label)
            ps[label] += np.log(p) # 对数概率,连乘变连加
    return ps.argmax() # 取概率最高的y输出

if __name__ == "__main__":
    x, y = GetData(fileName='WatermelonOriginalWithContinuous.txt')
    prob = ComputePostProbability(DataX=x, DataY=y, Col=0, Value=0, Y=0)

    prob_continuous = ComputePostProbabilityContinuous(DataX=x, DataY=y, Col=6, Value=0.697, Y=0)

    a = predict(x[0], x, y)

    correct = 0
    for xi, yi in zip(x, y):
        if predict(xi, x, y) == yi:
            correct += 1

    print(correct / x.shape[0])