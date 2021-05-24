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
    X_trainData = DataX[:trainDataLen]
    Y_trainData = DataY[:trainDataLen]
    X_testData = DataX[trainDataLen:]
    Y_testData = DataY[trainDataLen:]
    return X_trainData, Y_trainData, X_testData, Y_testData

def ComputeEntropy(feature):
    """
    计算熵
    :param feature:
    :return:
    """
    entropy = 0
    nums = len(feature) # 给定的某个特征下，所有的样本数
    bincounts = np.bincount(feature) # 统计featrue数组中每个数字出现的次数
    for count in bincounts:
        if count == 0 : continue # 如果特征出现的次数为0的话，直接跳过熵的计算
        prob = count / nums
        entropy -= prob * np.log2(prob) # 熵 = p * log(p) * -1
    return entropy

def ComputeGain(DataX, DataY, feature_index):
    """
    计算某一列，也就是某一个特征的信息增益
    :param DataX:
    :param DataY:
    :param feature_index:
    :return:
    """
    if DataY.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        DataY = np.expand_dims(DataY, axis=1)
    DataXY = np.concatenate((DataX, DataY), axis=1)
    feature_entropy = 0
    for feature in set(DataXY[:, feature_index]):
        DataByfeature = DataXY[DataXY[:, feature_index] == feature]
        prob = len(DataByfeature) / len(DataXY) # 特征feature_index下的特征子集出现的概率
        entropy = ComputeEntropy(DataByfeature[:, -1]) # 求数据子集下标签y的熵
        feature_entropy += prob * entropy #这个feature_index的熵,等于这个式子的累计
    # 计算信息增益，依据feature_index切分数据后，熵能下降多少，越大越好
    gain = ComputeEntropy(DataXY[:, -1]) - feature_entropy

    # 用这个就是id3决策树,他倾向于选择可取值多的列
    return gain

def GetMaxGain_FeatureIndex(DataX, DataY):
    """
    计算信息增益最大的那个特征，并返回
    :param DataX:
    :param DataY:
    :return:
    """
    BestFeatureIndex = -1
    BestGain = 0
    for feature_index in range(0, DataX.shape[1]):
        gain = ComputeGain(DataX, DataY, feature_index)
        # print("feature index {} , Gain {}".format(feature_index, gain))
        if gain > BestGain:
            BestFeatureIndex = feature_index
            BestGain = gain
    return BestFeatureIndex

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

def CreateChildren(DataX_train, DataY_train, DataX_test, DataY_test, ParentNode):
    if DataY_train.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        DataY_train = np.expand_dims(DataY_train, axis=1)
    DataXY = np.concatenate((DataX_train, DataY_train), axis=1)

    if DataY_test.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        DataY_test = np.expand_dims(DataY_test, axis=1)
    DataXY_test = np.concatenate((DataX_test, DataY_test), axis=1)

    for feature_index in np.unique(DataXY[:, ParentNode.col]):
        sub_data = DataXY[DataXY[:,ParentNode.col] == feature_index] #首先根据父节点col列的取值分割数据
        sub_data_test = DataXY_test[DataXY_test[:, ParentNode.col] == feature_index]

        SubData_UniqueY = np.unique(sub_data[:, -1])

        # 如果所有的y都是一样的,说明是个叶子节点
        # 如果分割后的测试正确率提升了,则分割,否则不分割
        if len(SubData_UniqueY) == 1 or pre_cut(DataX_train=sub_data[:,:-1], DataY_train=sub_data[:,-1], DataX_test=sub_data_test[:,:-1], DataY_test=sub_data_test[:,-1]):
            ParentNode.children[feature_index] = Leaf(SubData_UniqueY[0])
            continue
        MaxFeatureIndex = GetMaxGain_FeatureIndex(DataX=sub_data[:,:-1], DataY=sub_data[:, -1])
        # 添加分支节点到父节点上
        ParentNode.children[feature_index] = Node(col=MaxFeatureIndex)

def main(DataX_train, DataY_train, DataX_test, DataY_test):
    MaxFeatureIndex = GetMaxGain_FeatureIndex(DataX=DataX_train, DataY=DataY_train)
    root = Node(MaxFeatureIndex)
    print(root)

    CreateChildren(DataX_train=DataX_train, DataY_train=DataY_train, DataX_test=DataX_test, DataY_test=DataY_test, ParentNode=root)
    print_tree(root)
    print('#---------------------------------------#')
    # if DataY.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
    #     DataY = np.expand_dims(DataY, axis=1)
    # DataXY = np.concatenate((DataX, DataY), axis=1)
    # DataXY_0_0 = DataXY[DataXY[:,0] == 0]
    # CreateChildren(DataX=DataXY_0_0[:,:-1], DataY=DataXY_0_0[:, -1], ParentNode=root.children[0])
    # print_tree(root)
    # print('#---------------------------------------#')
    # DataXY_0_1 = DataXY[DataXY[:, 0] == 1]
    # CreateChildren(DataX=DataXY_0_1[:, :-1], DataY=DataXY_0_1[:, -1], ParentNode=root.children[1])
    # print_tree(root)

    # print('#---------------------------------------#')
    # # 继续创建,0=1,1=1的下一层
    # DataXY_0_1_and_1_1 = DataXY_0_1[DataXY_0_1[:, 1] == 1]
    # CreateChildren(DataX=DataXY_0_1_and_1_1[:,:-1], DataY=DataXY_0_1_and_1_1[:,-1], ParentNode=root.children[1].children[1])
    # print_tree(root)

    return root

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

def pre_cut(DataX_train, DataY_train, DataX_test, DataY_test):
    """
    判断是否要分割节点
    :param DataX_train:
    :param DataY_train:
    :param DataX_test:
    :param DataY_test:
    :return:
    """
    # 计算不分割前的测试正确率
    VoteY = np.bincount(DataY_train).argmax()
    pre_correct = np.sum(VoteY == DataY_test)

    if DataY_train.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        DataY_train = np.expand_dims(DataY_train, axis=1)
    DataXY_train = np.concatenate((DataX_train, DataY_train), axis=1)

    if DataY_test.shape.__len__() == 1: # 如果label的shape是一维的，就增加一个纬度
        DataY_test = np.expand_dims(DataY_test, axis=1)
    DataXY_test = np.concatenate((DataX_test, DataY_test), axis=1)

    # 求分割列
    BestFeatureIndex = GetMaxGain_FeatureIndex(DataX=DataX_train, DataY=DataY_train)

    # 计算分割后的正确率
    after_correct = 0
    for feature_index in np.unique(DataXY_train[:, BestFeatureIndex]):
        sub_data_train = DataXY_train[DataXY_train[:, BestFeatureIndex] == feature_index]
        sub_data_test = DataXY_test[DataXY_test[:, BestFeatureIndex] == feature_index]
        # 取y
        sub_data_train_y = sub_data_train[:, -1]
        sub_data_test_y = sub_data_test[:, -1]
        # 求众数
        sub_vote_y = np.bincount(sub_data_train_y).argmax()
        after_correct += np.sum(sub_vote_y == sub_data_test_y)

    # 如果分割后的测试正确率提升了,则分割,否则不分割
    return pre_correct >= after_correct

if __name__ == "__main__":
    Data_X, Data_Y = GetData(fileName='WatermelonOriginal.txt')
    trainData_X, trainData_Y, testData_X, testData_Y = split_Data(Data_X, Data_Y, ratio=0.6)
    print("label y entropy is : {}".format(ComputeEntropy(trainData_Y))) # 计算一下标签y的熵
    print("feature index=0 entropy is : {}".format(ComputeGain(trainData_X, trainData_Y, 0))) # 计算一下特征index为0的信息增益
    print("Best FeatureIndex Gain {} ".format(GetMaxGain_FeatureIndex(trainData_X, trainData_Y)))

    print(Node(0))
    print(Leaf(1))

    print_tree(Node(0))

    root = main(DataX_train=trainData_X, DataY_train=trainData_Y, DataX_test=testData_X, DataY_test=testData_Y)

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

    # import pickle
    # # 序列化保存下来,后面剪枝用
    # with open('tree.dump', 'wb') as fr:
    #     pickle.dump(root, fr)
