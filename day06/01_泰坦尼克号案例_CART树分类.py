"""
案例：演示CART分类回归决策树的分类功能
"""

# 导包
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1.加载数据
data = pd.read_csv('./data/titanic_train.csv')
# data.info()

# 2.数据预处理
# 2.1提取特征和标签
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# 2.2发现Age列有缺失值，用该列的平均值做填充
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3查看处理后的数据集
# x.info()

# 2.4针对Sex列，进行one-hot编码处理
x = pd.get_dummies(x, columns=['Sex'])
# x.info()

# 2.5划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3.特征工程

# 4.模型训练
# 参数max_depth=10：绘制的决策树结构最多有10层
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train, y_train)

# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为：{y_pred}')

# 6.模型评估
print(f'分类评估报告: \n {classification_report(y_test, y_pred)}')

# 7.绘制决策树图
plt.figure(figsize=(30, 20))    # 设置图片大小
# 参1：模型对象； 参2：是否用颜色填充； 参3：绘制的决策树结构，最多10层
plot_tree(estimator, filled=True, max_depth=10)
plt.savefig('./data/my_titanic.png')
plt.show()