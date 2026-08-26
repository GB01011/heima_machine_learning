"""
案例:
    基于用户的年收入和消费指数,根据用户的相似性进行聚类
"""

# 导包
# import os
# os.environ['OMP_NUM_THREADS'] = '4'     # 设置OMP程序运行时使用的线程数

import matplotlib
matplotlib.use('TkAgg')   # 绘图之后单独新窗口显示图

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import pandas as pd

# 1.定义函数，找聚类的质心数（k值）
def dm01_find_k():
    # 1.加载数据集
    df = pd.read_csv('./data/customers.csv')
    # df.info()
    # print(df.head())

    # 2.定义sse_list, sc_list, 记录不同k值的评估效果
    sse_list = []   # sse：只考虑簇内，越小越好
    sc_list = []    # sc：考虑簇内和簇间，越大越好

    # 抽取特征
    x = df.iloc[:, 3:5]
    print(x)

    # 3.定义for循环，测试不同k值的评估效果
    for k in range(2, 20):
        # 3.1.创建（KMeans模型）对象
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2.模型训练
        estimator.fit(x)
        # 3.3.模型预测
        y_pred = estimator.predict(x)
        # 3.4.分别把评分添加到对应的列中
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x, y_pred))

    # 4.绘制折线图，看看k纸哪个更好
    plt.figure(figsize=(10, 10))
    plt.plot(range(2, 20), sse_list, label='SSE')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(range(2, 20), sc_list, label='SC')
    plt.show()

# 2.定义函数，实现模型训练，模型预测，模型评估
def dm02_train_predict_evaluate():
    # 1.加载数据集
    df = pd.read_csv('./data/customers.csv')
    # 2.提取特征
    x = df.iloc[:, 3:5]
    print(x.head())
    # 3.模型训练
    # k=5是刚才通过sse+肘方法,sc轮廓系数获取出来的
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=23)
    estimator.fit(x)
    # 4.模型预测
    y_pred = estimator.predict(x)
    print(y_pred)
    # 5.绘制5个簇的样本点 ———— 散点图
    plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1])     # 0号簇
    plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1])     # 1号簇
    plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1])     # 2号簇
    plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1])     # 3号簇
    plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1])     # 4号簇

    # 6.绘制5个簇的质心 ———— 散点图
    print(estimator.cluster_centers_)   # 5个质心的坐标
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1])

    # 7.设置标题，x轴，y轴标签
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


# 3.测试
if __name__ == '__main__':
    # dm01_find_k()
    dm02_train_predict_evaluate()