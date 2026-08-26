"""
案例：演示逻辑回归模型实现癌症预测
逻辑回归模型介绍：
    概述：
        属于有监督学习，即有特征、有标签，且标签是离散的
        主要适用于二分类
    原理：
        把线性回归处理后的预测值通过Sigmoid激活函数映射到[0,1] ———— 概率：基于自定义的阈值结合概率来分类
    损失函数：
        极大似然估计函数的负数形式

回顾：
    机器学习的步骤：
        1.加载数据
        2.数据预处理
        3.特征工程（特征提取、特征与处理、特征降维、特征选择、特征组合）
        4.模型训练
        5.模型预测
        6.模型评估
"""

# 导包
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression     # 逻辑回归模型
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.model_selection import train_test_split    # 切分训练集和测试集
from sklearn.metrics import accuracy_score              # 模型评估

# 1.加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info() # 查看数据信息

# 2.数据预处理
# 2.1把异常值（？）替换成np.nan；参1：要被替换的值； 参2：用来替换的值； 参3：是否替换原数据，默认为False
data.replace('?',np.nan,inplace = True)
# 2.2缺失值处理————删除
data.dropna(axis = 0, inplace=True) # axis=0表示行，删除包含缺失值的行
# 2.3打印处理后的信息
# data.info()

# 3.特征工程（特征提取、特征与处理...）
# 3.1特征提取之提取特征和标签
x = data.iloc[:, 1:-1]  # 按照行号、列索引获取数据， :表示所有行，     1:-1表示从第1列到最后1列
# y = data.iloc[:, -1]    # 获取最后一列
# y = data['Class']       # 获取最后一列，效果同上
y = data.Class          # 获取最后一列，效果同上

# 3.2查看特征和标签
print(x[:5])
print(y[:5])
print(x.shape, y.shape)

# 3.3切割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3.4特征工程：标准化
# 3.4.1创建标准化对象
transfer = StandardScaler()
# 3.4.2对训练集进行标准化，训练+标准化
x_train = transfer.fit_transform(x_train)
# 3.4.3对测试集进行标准化，标准化
x_test = transfer.transform(x_test)

# 4.模型训练
# 4.1创建模型对象————逻辑回归模型
estimator = LogisticRegression()
# 4.2模型训练
estimator.fit(x_train, y_train)

# 5.模型预测
y_pre = estimator.predict(x_test)
print(f'预测值为：{y_pre}')

# 6.模型评估
# 正确率（准确率），公式为：预测对的 / 样本总数
print(f'预测前评估正确率：{estimator.score(x_test, y_test)}')    # 测试集的特征、标签
print(f'预测后评估正确率：{accuracy_score(y_test, y_pre)}')      # 测试集的标签、预测值

# 思考：逻辑回归模型能用准确率进行评估吗？
# 答：可以但是不精准；因为逻辑回归是主要用于二分类，即A类还是B类，不能说97%的A类、3%的B类
# 所以要通过混淆矩阵来评测，即精准率，召回率，F1值（F1-score），ROC曲线，AUC值