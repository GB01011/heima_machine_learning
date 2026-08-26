"""
案例：演示特征预处理 ———— 归一化操作
回顾：特征工程的目的和步骤
    目的：
        利用专业的背景知识 和 技巧处理数据，用于提升模型的性能
    步骤：
        1.特征提取
        2.特征预处理（归一化，标准化）
        3.特征降维
        4.特征选择
        5.特征组合

特征预处理之 归一化介绍：
    目的：
        防止因为量纲（即单位）问题，导致特征列的方差值相差较大，影响模型的最终效果
        所以通过公式把各列的值映射到【0，1】区间（默认是【0，1】，也可以自己设置为其他区间值）
    公式：
        x' = （当前值 - 该列最小值） / （该列最大值 - 该列最小值）
        x'' = x' * （mx - mi） + mi
    公式解释：
        x' ———— 基于公式算出来的结果
        x'' ———— 最终的结果
        mx ———— 区间的最大值
        mi ———— 区间的最小值
    弊端：
        容易受到最大值和最小值的影响，所以一般用来处理小数据集
"""

# 导包
# preprocessing是预处理的包
# MinMaxScaler是归一化的包
# StandardScaler是标准化的包
from sklearn.preprocessing import MinMaxScaler

# 1.准备数据集（归一化之前的原数据）
x_train = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]

# 2.创建归一化对象
# 参数feature_range表示生成范围，默认为：【0，1】，如果就是这个区间则参数可以省略不写
# transfer = MinMaxScaler(feature_range=(3,5))
# feature_range是缩放区间，即[min,max]区间
transfer = MinMaxScaler()

# 3.对原数据进行归一化操作
# fit_transform(X)将特征进行归一化缩放，即训练并转换，适用于训练集；
# transform(X)则适用于测试集
x_train_new = transfer.fit_transform(x_train)

# 4.打印处理后的数据
print('归一化后的数据集为: \n')
print(x_train_new)