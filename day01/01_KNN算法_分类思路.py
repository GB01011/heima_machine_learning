"""
KNN算法介绍 (K Nearest Neighbors)，K近邻算法
    原理：
        基于欧氏距离（或者其他距离计算方式）计算 测试集 和 训练集之间的距离，然后根据距离进行升序排列，找到最近的K个样本
        基于K个样本投票，票数多的就作为最终预测结果————分类问题
        基于K个样本计算平均值，作为最终预测结果————回归问题
    实现思路：
        1.分类问题
            适用于：有特征、有标签，标签不连续（离散）
        2.回归问题
            适用于：有特征、无标签，标签连续
    KNN算法分类问题的思路：
        1.计算测试集和每个训练集的样本之间的距离
        2.基于距离进行升序排列
        3.找到最近的K个样本
        4.对K个样本进行投票
        5.票数多的结果作为最终的预测结果
    代码实现思路：
        1.导包
        2.准备数据集（测试集 和 训练集）
        3.创建（KNN 分类模型）模型对象
        4.模型训练
        5.模型预测
"""

# 1.导包
# sklearn：机器学习库
# neighbors近邻
# KNeighborsClassifier分类
# KNeighborsRegressor回归
from sklearn.neighbors import KNeighborsClassifier

# 2.准备数据集（测试集 和 训练集）
# train：训练集
# test：测试集
# neighbors：最近邻的邻居数
x_train = [[0], [1], [2], [3]]  # 训练集的特征，由于特征可以有多个，所以是一个二维数组
y_train = [0, 0, 1, 1]  # 训练集的标签，由于标签是离散的，所以是一个一维数组
x_test = [[5]]  # 测试集的特征

# 3.创建（KNN 分类模型）模型对象
# estimator：估计器，模型对象，也可以用变量名model
estimator = KNeighborsClassifier(n_neighbors=2)

# 4.模型训练（拟合）
# 传入：训练集的特征数据，训练集的标签数据
estimator.fit(x_train,y_train)

# 5.模型预测
# 传入：测试集的特征数据，获取到 预测结果（y_test / y_predict）
y_predict = estimator.predict(x_test)

# 6.打印预测结果
# 运算用欧氏距离（也可以理解为勾股定理）：对应维度的差值平方和。然后开根号
# 先用x_test与x_train的数据分别进行差值平方和然后开根号
print(f'预测值为：{y_predict}')
"""
数据详情：
x_train     y_train
0           0
1           0
2           1
3           1

计算结果：
计算结果        对应值
5               0
4               0
3               1
2               1

基于结算结果（即距离）进行升序排列：
计算结果    x_train     y_train
2           3           1
3           2           1
4           1           0
5           0           0

由于K = 2，所以取最近的两个，对应的结果为2、3
取距离最近的，则取结果2，对应的值为1
"""