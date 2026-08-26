"""
案例：
    演示KMeans聚类算法入门案例

Kmeans简介：
    它属于无监督学习，即有特征、无标签，根据样本间的相似性进行划分
    所谓的相似性可以理解为就是距离，如：欧式距离，曼哈顿（城市街区）距离，切比雪夫距离，闵式距离...
"""

# 导包
import matplotlib
matplotlib.use('TkAgg')

from sklearn.cluster import KMeans  # 聚类的API，采用指定质心来分簇
import matplotlib.pyplot as plt # 绘图
from sklearn.datasets import make_blobs # 默认会按照高斯分布（正态分布）生成数据集，只需要指定均值，标准差
from sklearn.metrics import calinski_harabasz_score # 评价指标，值越大，聚类效果越好

# 1.准备数据集
# 参1：样本数量； 参2：样本特征数量（2列）， 参3：样本标签数量（3类）， 参4：标准差， 参5：随机种子
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]], cluster_std=[0.4, 0.2, 0.3, 0.4], random_state=23)
# print(x)
# print(y)

# 2.绘制上述的图形
# 参1：横坐标， 参2：纵坐标， 参3：颜色
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# 3.创建KMeans对象
# 参1：聚类数量， 参2：随机种子
estimator = KMeans(n_clusters=4, random_state=23)

# 4.模型训练和预测
y_pred = estimator.fit_predict(x)   # 预测值

# 5.绘制预测结果
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# 6.评价指标
print(f'评价指标（评分）：{calinski_harabasz_score(x, y_pred)}') # 值越大越好