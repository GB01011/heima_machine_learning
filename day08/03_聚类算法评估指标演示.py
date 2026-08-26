"""
案例：
    演示聚类算法的评估指标，即SSE + 肘方法，SC轮廓系数法，CH轮廓系数法

聚类算法的评估指标：
    思路1：SSE + 肘方法
        SSE：
            概述：
                所有簇的所有样本到该簇质心的误差的平方和
            特点：
                随着K值的增加，SSE值会逐渐减少
            目标：
                SSE值越小，代表簇内样本越聚集，内聚程度越高
        肘方法：
            K值增大，SSE值会随之减小，下降梯度陡然变缓的时候，那个K值就是最佳值

    思路2：SC轮廓系数
        考虑簇内 —— 聚集程度，越小越好
        考虑簇外 —— 分离程度，越大越好

    思路3：CH轮廓系数
        考虑簇内 —— 聚集程度，越小越好
        考虑簇外 —— 分离程度，越大越好
        考虑K值 —— K值越小，代表簇内样本越聚集，内聚程度越高
"""

# 导包
# import os
# os.environ['OMP_NUM_THREADS'] = '4'     # 设置OMP程序运行时使用的线程数

import matplotlib
matplotlib.use('TkAgg')   # 绘图之后单独新窗口显示图

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # 造数据
from sklearn.metrics import calinski_harabasz_score, silhouette_score


# 1.定义函数，演示SSE+肘方法
def dm01_sse():

    # 1.定义sse列表，记录每个k值的SSE值
    sse_list = []

    # 2.生成数据集
    # 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3.for循环遍历，获取到每个k值，计算其对应的sse值，并添加到sse_list列表中
    for k in range(1, 100):
        # 3.1创建KMeans对象
        # 参1：簇的数量， 参2：最大迭代次数， 参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2模型训练
        estimator.fit(x)
        # 3.3模型预测
        # 3.4获取到每个簇的sse值
        sse_value = estimator.inertia_
        # 3.5将每个k值对应的sse值添加到sse_list列表中
        sse_list.append(sse_value)

    # 4.绘制SSE曲线 ———— 数据可视化
    print(sse_list)
    # 4.1创建画布，指定画布尺寸
    plt.figure(figsize=(20, 10))
    # 4.2设置标题
    plt.title('sse value')
    # 4.3设置x轴的刻度
    plt.xticks(range(0, 100, 3))
    # 4.4添加x轴，y轴的标签
    plt.xlabel('k')
    plt.ylabel('sse')
    # 4.5绘制网格
    plt.grid()
    # 4.6绘制折线图
    # 参1:k值, 参2:该k值对应的sse值
    plt.plot(range(1, 100), sse_list)
    # 4.7显示图形
    plt.show()

# 2.定义函数，演示SC轮廓系数法
def dm02_sc():

    # 1.定义sc列表，记录每个k值的sc值
    sc_list = []

    # 2.生成数据集
    # 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3.for循环遍历，获取到每个k值，计算其对应的sc值，并添加到sc_list列表中
    for k in range(2, 100): # 考虑簇外，至少需要两个簇，所以从2开始遍历
        # 3.1创建KMeans对象
        # 参1：簇的数量， 参2：最大迭代次数， 参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2模型训练
        estimator.fit(x)
        # 3.3模型预测
        y_pred = estimator.predict(x)
        # 3.4获取到每个簇的sc值
        sc_value = silhouette_score(x, y_pred)
        # 3.5将每个k值对应的sc值添加到sc_list列表中
        sc_list.append(sc_value)

    # 4.绘制sc曲线 ———— 数据可视化
    print(sc_list)
    # 4.1创建画布，指定画布尺寸
    plt.figure(figsize=(20, 10))
    # 4.2设置标题
    plt.title('sc value')
    # 4.3设置x轴的刻度
    plt.xticks(range(0, 100, 3))
    # 4.4添加x轴，y轴的标签
    plt.xlabel('k')
    plt.ylabel('sc')
    # 4.5绘制网格
    plt.grid()
    # 4.6绘制折线图
    # 参1:k值, 参2:该k值对应的sc值
    plt.plot(range(2, 100), sc_list)
    # 4.7显示图形
    plt.show()

# 3.定义函数，演示CH轮廓系数法
def dm03_ch():

    # 1.定义ch列表，记录每个k值的ch值
    ch_list = []

    # 2.生成数据集
    # 参1：样本数量， 参2：特征数量， 参3：4个簇， 参4：4个簇的std标准差， 参5：固定随机种子
    x, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
        cluster_std=[0.4, 0.2, 0.2, 0.2],
        random_state=23
    )

    # 3.for循环遍历，获取到每个k值，计算其对应的ch值，并添加到ch_list列表中
    for k in range(2, 100): # 考虑簇外，至少需要两个簇，所以从2开始遍历
        # 3.1创建KMeans对象
        # 参1：簇的数量， 参2：最大迭代次数， 参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2模型训练
        estimator.fit(x)
        # 3.3模型预测
        y_pred = estimator.predict(x)
        # 3.4获取到每个簇的ch值
        ch_value = calinski_harabasz_score(x, y_pred)
        # 3.5将每个k值对应的ch值添加到ch_list列表中
        ch_list.append(ch_value)

    # 4.绘制ch曲线 ———— 数据可视化
    print(ch_list)
    # 4.1创建画布，指定画布尺寸
    plt.figure(figsize=(20, 10))
    # 4.2设置标题
    plt.title('ch value')
    # 4.3设置x轴的刻度
    plt.xticks(range(0, 100, 3))
    # 4.4添加x轴，y轴的标签
    plt.xlabel('k')
    plt.ylabel('ch')
    # 4.5绘制网格
    plt.grid()
    # 4.6绘制折线图
    # 参1:k值, 参2:该k值对应的ch值
    plt.plot(range(2, 100), ch_list)
    # 4.7显示图形
    plt.show()

# 4.
if __name__ == '__main__':
    # dm01_sse()
    # dm02_sc()
    dm03_ch()