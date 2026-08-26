"""
案例:演示KNN算法识别图片,即:手写数字识别案例

介绍:
    每张图片都是由28 * 28 像素组成的,即:csv文件中每一行都有784个像素点,表示图片(每个像素)的颜色
    最终构成图像
"""

# 导包
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib   # 模型训练完成之后保存模型,之后可以直接使用训练好的模型不用重新去运行一遍模型的训练
from collections import Counter     # 计数,去重统计
from sklearn.metrics import accuracy_score                          # 模型评估，计算模型预测的准确率

# 忽略警告
import warnings
warnings.filterwarnings('ignore',module='sklearn')   # 将sklearn这个模块下可能会出现的警告忽略

# 1.定义函数,接收用户传入的索引,展示该索引对应的图片
def show_digit(idx):
    # 1.读取数据,获取到原数据
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)

    # 2.判断传入的索引是否越界
    if idx < 0 or idx > len(df) - 1:
        print('索引越界!')
        return

    # 3.如果执行到这里则说明没有越界,正常获取数据
    x = df.iloc[:,1:]       # 第一列是标签,第二列开始往后
    # 都是特征
    y = df.iloc[:,0]        # 拿到第一列,即标签

    # 4.查看用户传入的索引对应的图片是几
    print(f'该图片对应的数字是:{y.iloc[idx]}')
    print(f'查看所有的标签的分布情况:{Counter(y)}')     # 将文件中的各个数据的总个数情况输出出来

    # 5.查看下用户传入的索引对应的图片的形状
    print(x.iloc[idx].shape)        # (784,) 我们要想办法把(784,)转换成(28,28)
    # print(x.iloc[idx].values)       # 具体的784个像素点

    # 6.把(784,)转换成(28,28)
    x = x.iloc[idx].values.reshape(28,28)
    # print(x)      # 28 * 28 像素点

    # 7.具体的绘制灰度图的动作
    plt.imshow(x,cmap = 'gray')     # cmap是灰度图
    plt.axis('off')                 # 不显示坐标轴
    plt.show()

# 2.定义函数,训练模型,并保存训练好的模型
def train_model():
    # 1.加载数据集
    df = pd.read_csv('./data/手写数字识别.csv')

    # 2.数据的预处理
    # 2.1拆分出特征列
    x = df.iloc[:,1:]       # 特征列
    # 2.2拆分出特征列
    y = df.iloc[:,0]        # 标签列
    # 2.3打印特征和标签的形状
    print(f'x的形状:{x.shape}')        # (42000,784)
    print(f'y的形状:{y.shape}')        # (42000,1)
    print(f'查看所有的标签的分布情况:{Counter(y)}')
    # 2.4对特征列(拆分前)进行归一化
    x = (x - 0) / (255 - 0)
    # 2.5拆分训练集和测试集
    # 参1:特征列 ; 参2:标签列 ; 参3:测试集的比例 ; 参4:随机种子; 参5:参考y值进行抽取,保持标签的比例(数据均衡,防止有一些数据没被抽取到)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22,stratify=y)

    # 3.模型训练
    # 3.1创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 3.2模型训练
    estimator.fit(x_train,y_train)

    # 4.模型评估
    print(f'准确率: {estimator.score(x_test,y_test)}')
    print(f'准确率: {accuracy_score(y_test,estimator.predict(x_test))}')   # 评估的另一方式

    # 5.保存模型
    # 参1：模型对象     参2：模型保存的路径
    joblib.dump(estimator,'./model/手写数字识别.pkl')     # pickle文件：Python（Pandas）独有的文件类型
    print('模型保存成功')

# 3.定义函数，测试模型
def use_model():
    # 1.加载图片
    x = plt.imread('./data/demo.png')   # 这里的x是指 28 * 28 个像素
    # 2.绘制图片
    plt.imshow(x,cmap='gray')           # 灰度图
    # plt.show()

    # 3.加载模型
    estimator = joblib.load('./model/手写数字识别.pkl')

    # 4.模型预测
    # 4.1查看数据集转换
    print(x.shape)                  # (28,28)
    print(x.reshape(1,784).shape)   # (1,784),因为csv文件数据是一行784列数据，所以把28*28的数据变成784的格式
    print(x.reshape(1,-1).shape)    # 效果等同于(1,784),-1表示有多少行就转成多少，一般写-1的这个，因为不用去实际知道有多少列

    # 4.2具体的转换动作，记得要进行归一化（因为上面训练模型的时候使用了归一化动作）
    # x = x.reshape(1,-1) / 255
    # 不过在这里不能进行归一化，因为可能会预测失败：读图的时候，像素值可能不是特别的精准，读取出来的像素数据可能会有偏差
    x = x.reshape(1,-1)     # 直接使用原始的读取到的像素值做预测

    # 4.3模型预测
    y_pre = estimator.predict(x)

    # 5.打印预测结果
    print(f'预测值为：{y_pre}')

# 4.测试
if __name__ == '__main__':
    # 绘制数字
    # show_digit(9)       # 传入的是9,对应的csv文件中的索引是11处的数据,为3

    # 训练模型并保存模型
    # train_model()

    # 模型预测（使用模型）
    use_model()