import os
import pickle
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
    X_trainData = DataX[:trainDataLen]
    Y_trainData = DataY[:trainDataLen]
    X_testData = DataX[trainDataLen:]
    Y_testData = DataY[trainDataLen:]
    return X_trainData, Y_trainData, X_testData, Y_testData

class Node():
    def __init__(self, col):
        """
        #创建节点和叶子对象,用来构建树
        :param col:
        """
        self.col = col
        self.children = {}

    def __str__(self):
        return 'Node col={}'.format(self.col)

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
        subfix = 'value=' + str(i)
        print_tree(node.children[i], prefix, subfix)

def after_cut(node, DataXY_train, DataXY_test):
    # 先计算不剪枝的正确率
    pre_correct = 0
    for split_value in np.unique(DataXY_test[:, node.col]):
        pre_y = DataXY_test[DataXY_test[:, node.col] == split_value][:, -1]
        pre_correct += np.sum(pre_y == node.children[split_value].y)

    # 计算剪切的测试正确率
    DataXY_train_Y = DataXY_train[:, -1]
    DataXY_test_Y = DataXY_test[:, -1]
    Vote_Y = np.bincount(DataXY_train_Y).argmax()
    after_correct = np.sum(Vote_Y == DataXY_test_Y)
    if after_correct >= pre_correct: # 如果剪枝之后准确率上升或不变,就剪切,否则不剪切
        return Leaf(y=Vote_Y)
    return node

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

def main(DataXY_train, DataXY_test):
    # 加载树
    with open('tree.dump', 'rb') as f:
        root = pickle.load(f)
    print_tree(root)

    # 先从最后一个节点开始切,也就是0=1,1=1。先把数据和测试数据筛选出来
    DataXY_train_0_1 = DataXY_train[DataXY_train[:, 0] == 1]
    DataXY_train_0_1_and_1_1 = DataXY_train_0_1[DataXY_train_0_1[:, 1] == 1]
    DataXY_test_0_1 = DataXY_test[DataXY_test[:, 0] == 1]
    DataXY_test_0_1_and_1_1 = DataXY_test_0_1[DataXY_test_0_1[:, 1] == 1]
    node = root.children[1].children[1] # 取节点
    root.children[1].children[1] = after_cut(node, DataXY_train_0_1_and_1_1, DataXY_test_0_1_and_1_1)
    print_tree(root)
    print('#-----#----' * 8)

    # 剪切0=1, 先把数据和测试数据筛选出来
    DataXY_train_0_1 = DataXY_train[DataXY_train[:, 0] == 1]
    DataXY_test_0_1 = DataXY_test[DataXY_test[:, 0] == 1]
    node = root.children[1] # 取节点
    root.children[1] = after_cut(node, DataXY_train_0_1, DataXY_test_0_1)
    print_tree(root)
    print('#-----#----' * 8)

    # 剪切0=0, 先把数据和测试数据筛选出来
    DataXY_train_0_0 = DataXY_train[DataXY_train[:, 0] == 0]
    DataXY_test_0_0 = DataXY_test[DataXY_test[:, 0] == 0]
    node = root.children[0]
    root.children[0] = after_cut(node, DataXY_train_0_0, DataXY_test_0_0)
    print_tree(root)
    print('#-----#----' * 8)

    # 剪切根节点
    root = after_cut(root, DataXY_train, DataXY_test)
    print_tree(root)
    return root

if __name__ == "__main__":
    Data_X, Data_Y = GetData(fileName='WatermelonOriginal.txt')
    trainData_X, trainData_Y, testData_X, testData_Y = split_Data(Data_X, Data_Y, ratio=0.6)

    if trainData_Y.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        trainData_Y = np.expand_dims(trainData_Y, axis=1)
    DataXY_train = np.concatenate((trainData_X, trainData_Y), axis=1)

    if testData_Y.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        testData_Y = np.expand_dims(testData_Y, axis=1)
    DataXY_test = np.concatenate((testData_X, testData_Y), axis=1)

    root = main(DataXY_train, DataXY_test)

    # 训练集上测试效果
    correct = 0
    for x, y in zip(trainData_X, trainData_Y):
        if pred(x, root) == y:
            correct += 1
    print(correct / len(trainData_X))
    print('-------------------------')

    # 测试集上测试效果
    correct = 0
    for x, y in zip(testData_X, testData_Y):
        if pred(x, root) == y:
            correct += 1
    print(correct / len(trainData_X))
    print('-------------------------')

