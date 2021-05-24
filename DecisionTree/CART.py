import os
import numpy as np
import pandas as pd

def GetData(fileName = 'WatermelonOriginal.txt'):
    Data_FilePath = os.path.dirname(os.getcwd()) + '/Data/' + fileName
    df = pd.read_csv(Data_FilePath, sep=',', header=None, names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6','y'])
    X_trainData = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
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

def ComputeGain(Data, col, value):
    """
    # 计算基尼系数
    :param Data:
    :param col:
    :param value:
    :return:
    """
    Gini = 0
    for symbol in ["Equal", "NotEqual"]:
        if symbol == "Equal":
            subData = Data[Data[:, col] == value]
        elif symbol == "NotEqual":
            subData = Data[Data[:, col] != value]
        else:
            print("There is Something error")
            break
        if len(subData) == 0: Gini += 1e20
        prob = len(subData) / len(Data) # 计算两部分数据各自出现的概率
        # 计算在两部分数据中,y的占比
        prob_y0 = np.sum(subData[:, -1] == 0) / len(subData)
        prob_y1 = np.sum(subData[:, -1] == 1) / len(subData)
        # 计算两边的基尼指数, gini = 数据出现的概率 * (1 - y0概率的平方 - y1概率的平方)
        Gini += prob * (1 - np.power(prob_y0, 2) - np.power(prob_y1, 2))
    return Gini

def GetMaxGini_FeatureIndex(Data):
    min_feature_index = None
    min_feature = None
    min_gini = 1e20

    for feature_index in range(0, Data.shape[1]-1): # 遍历所有的列,最后一列是y,不需要计算。
        for feature in set(Data[:, feature_index]): # 遍历所有取值。
            len_feature = np.sum(Data[:, feature_index] == feature)
            if len_feature == len(Data) or len_feature == 0: # 如果一个字段只有一个值的话,就不能切
                continue
            # 信息增益,就是切分数据后,熵值能下降多少,这个值越大越好
            Gini = ComputeGain(Data=Data, col=feature_index, value=feature)
            # 信息增益最大的列,就是要被拆分的列
            if Gini < min_gini:
                min_feature_index = feature_index
                min_feature = feature
                min_gini = Gini
    return min_feature_index, min_feature

class Node():
    def __init__(self, col, value):
        self.col = col
        self.value = value
        self.children = {}

    def __str__(self):
        return 'Node col={} value={}'.format(self.col, self.value)

class Leaf():
    def __init__(self, y):
        self.y = y

    def __str__(self):
        return 'Leaf y={}'.format(self.y)

def print_tree(node, prefix='', subfix=''):
    """
    # 打印树的方法
    :param node:
    :param prefix:
    :param subfix:
    :return:
    """
    prefix += '-' * 4
    print(prefix, node, subfix)
    if isinstance(node, Leaf):
        return
    for i in node.children:
        subfix = 'symbol=' + str(i)
        print_tree(node.children[i], prefix, subfix)

def CreateChildren(Data, ParentNode):
    for symbol in ["Equal", "NotEqual"]:
        # 首先根据父节点col列的取值分割数据
        if symbol == "Equal":
            subData = Data[Data[:, ParentNode.col] == ParentNode.value]
        elif symbol == "NotEqual":
            subData = Data[Data[:, ParentNode.col] != ParentNode.value]
        else:
            print("There is Something error")
            break
        UniqueY = np.unique(subData[:, -1])
        if len(UniqueY) == 1: # 如果所有的y都是一样的,说明是个叶子节点
            ParentNode.children[symbol] = Leaf(UniqueY[0])
            continue
        # 否则,是个分支节点,计算最佳切分列
        split_col, split_val = GetMaxGini_FeatureIndex(Data=subData)
        # 添加分支节点到父节点上
        ParentNode.children[symbol] = Node(col=split_col, value=split_val)

def pred(DataX, node):
    """
    #预测方法,测试
    :param DataX:
    :param node:
    :return:
    """
    col_value = DataX[node.col]
    node = node.children[col_value]
    if isinstance(node, Leaf):
        return node.y

    return pred(DataX, node)

def main(DataXY_train):
    # 先在所有数据上求最大信息增益的列
    feature_index, feature_value = GetMaxGini_FeatureIndex(DataXY_train)
    root = Node(col=feature_index, value=feature_value)
    print_tree(root)
    print('#---------------------------------------#')

    CreateChildren(Data=DataXY_train, ParentNode=root)
    print_tree(root)
    print('#---------------------------------------#')

    DataXY_train_3_eq_0 = DataXY_train[DataXY_train[:, 3] == 0]
    DataXY_train_3_neq_0 = DataXY_train[DataXY_train[:, 3] != 0]
    CreateChildren(Data=DataXY_train_3_eq_0, ParentNode=root.children['Equal'])
    CreateChildren(Data=DataXY_train_3_neq_0, ParentNode=root.children['NotEqual'])
    print_tree(root)
    print('#---------------------------------------#')

    DataXY_train_3_eq_0_and_5_neq_0 = DataXY_train_3_eq_0[DataXY_train_3_eq_0[:, 5] != 0]
    CreateChildren(Data=DataXY_train_3_eq_0_and_5_neq_0, ParentNode=root.children['Equal'].children['NotEqual'])
    DataXY_train_3_neq_0_and_0_eq_1 = DataXY_train_3_neq_0[DataXY_train_3_neq_0[:, 0] == 1]
    CreateChildren(Data=DataXY_train_3_neq_0_and_0_eq_1, ParentNode=root.children['NotEqual'].children['Equal'])
    print_tree(root)
    print('#---------------------------------------#')
    return root

if __name__ == "__main__":
    Data_X, Data_Y = GetData(fileName='WatermelonOriginal.txt')
    trainData_X, trainData_Y, testData_X, testData_Y = split_Data(Data_X, Data_Y, ratio=0.6)

    if trainData_Y.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        trainData_Y = np.expand_dims(trainData_Y, axis=1)
    DataXY_train = np.concatenate((trainData_X, trainData_Y), axis=1)

    if testData_Y.shape.__len__() == 1:
        testData_Y = np.expand_dims(testData_Y, axis=1)
    DataXY_test = np.concatenate((testData_X, testData_Y), axis=1)

    # 计算第0列，值为0的基尼系数
    print("ComputeGain of feature_index = 0 : {}".format(ComputeGain(Data=DataXY_train, col=0, value=0)))
    # 返回基尼系数最大的那个特征index和其对应的特征值
    print("MaxGain feature_index and value {}".format(GetMaxGini_FeatureIndex(DataXY_train)))

    root = main(DataXY_train=DataXY_train)

