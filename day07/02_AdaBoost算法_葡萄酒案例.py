"""
案例：
    演示AdaBoost算法之葡萄酒案例
AdaBoost算法介绍：
    它属于Boosting思想，即：串行执行，每次使用全部样本，最后加权投票
    原理：
        1.使用全部样本，通过决策树模型（第一个弱分类器）进行训练，获取结果
            思路：
                预测正确 ———— 权重下降
                预测错误 ———— 权重上升
        2.把第一个弱分类器的处理结果交给第二个弱分类器进行训练，获取结果
            思路：
                预测正确 ———— 权重下降
                预测错误 ———— 权重上升
        3.以此类推，串行执行，直到获取最终结果
"""

# 导包
import pandas as pd
from sklearn.preprocessing import LabelEncoder          # 标签编码器
from sklearn.model_selection import train_test_split    # 训练集、测试集分割
from sklearn.tree import DecisionTreeClassifier         # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost分类器 ———— 集成学习Boosting思想
from sklearn.metrics import accuracy_score              # 模型评估 ———— 正确率

# 1.读取数据集
df_wine = pd.read_csv('./data/wine0501.csv')
# df_wine.info()
# print(df_wine['Class label'].unique())  # [1, 2, 3]葡萄酒类别有3种，但是决策树只能识别二叉树

# 2.数据预处理
# 2.1从标签列（Class label）中过滤掉1类别，剩下2，3类别
df_wine = df_wine[df_wine['Class label'] != 1]
# print(df_wine['Class label'].unique())

# 2.2获取特征列和标签列
x = df_wine[['Alcohol', 'Hue']]     # 酒精 和 色泽
y = df_wine['Class label']          # 标签列

# 2.3打印数据
print(x[:5])
print(y[:5])

# 2.4通过标签编码器把标签列转换为数值列
le = LabelEncoder()
y = le.fit_transform(y)
print(y)        # 将[2, 3] 转化为 [0, 1]

# 2.5训练集、测试集分割
# stratify=y：抽取数据时尽量参考y轴
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

# 3.特征工程，此处略

# 4.模型训练,预测,评估
# 场景1:单一决策树 -- 充当弱分类器
# 4.1创建模型对象
estimator1 = DecisionTreeClassifier(max_depth=3)
# 4.2模型训练
estimator1.fit(x_train, y_train)
# 4.3模型预测
y_pre1 = estimator1.predict(x_test)
print(f'单一决策树预测结果:{y_pre1}')
# 4.4模型评估
print(f'单一决策树预测正确率:{accuracy_score(y_test, y_pre1)}')

# 场景2：AdaBoost ———— 集成学习，CART树，200棵
# 4.1创建模型对象
# 参1：弱分类器（决策树对象）， 参2：弱分类器个数， 参3：学习率
estimator2 = AdaBoostClassifier(estimator=estimator1, n_estimators=200, learning_rate=0.1)
# 4.2训练模型
estimator2.fit(x_train, y_train)
# 4.3模型预测
y_pre2 = estimator2.predict(x_test)
print(f'AdaBoost集成学习预测结果:{y_pre2}')
# 4.4模型评估
print(f'AdaBoost集成学习预测正确率:{accuracy_score(y_test, y_pre2)}')