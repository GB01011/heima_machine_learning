"""
案例：
    演示通过朴素贝叶斯算法实现商品评论情感分析，即：好评、差评...

朴素贝叶斯介绍：
    概述：
        贝叶斯：仅仅依赖概率就可以进行分类的一种机器学习算法
        朴素：不考虑特征之间的关联性，即：特征间都是互相独立的
            原始：P(AB) = P(A) * P(B|A) = P(B) * P(A|B)
            加入朴素后：P(AB) = P(A) * P(B)
        注意：
            分词需要用jieba包
"""

# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba    # 分词包
from sklearn.feature_extraction.text import CountVectorizer # 词频统计包，把评论内容转成词频矩阵
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB   # 朴素贝叶斯对象

# 1.读取文件，获取到原始数据
df = pd.read_csv('./data/书籍评价.csv', encoding='gbk')
# df.info()

# 2.数据预处理
# 2.1添加labels列，充当标签列  好评 ———— 1， 差评 ———— 0
df['labels'] = np.where(df['评价'] == '好评', 1, 0)
# df.info()
# print(df)

# 2.2抽取labels列作为标签
y = df['labels']

# 2.3对用户的评论信息做切词
# 数据格式：[[第一条评论切词1，切词2，切词3...], [第二条评论切词1，切词2，切词3...]...]
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
# 数据格式：['第一条评论切词1，切词2，切词3...', ‘第二条评论切词1，切词2，切词3...’, ...]
print(comment_list)

# 演示字符串的join()函数用法
# my_list = ['aa', 'bb', 'cc']
# print(','.join(my_list))

# 2.4加载停用词列表，即txt中记录的词，不需要参与模型训练、预测，要被删除的词例如的，啊，哈，...
with open('./data/stopwords.txt', 'r', encoding='utf-8') as src_f:
    # 2.4.1一次读取所有行
    stopwords_list = src_f.readlines()
    # 2.4.2删除最后的'\n'
    stopwords_list = [line.strip() for line in stopwords_list]
    # 2.4.3对停用词列表去重
    stopwords_list = list(set(stopwords_list))
    print(stopwords_list)

# 2.5创建向量化对象，从评论切词列表（comment_list）中删除停用词，并且统计词频（单词矩阵）
transfer = CountVectorizer(stop_words=stopwords_list)   # 参数：停用词列表

# 2.6统计词频矩阵
x = transfer.fit(comment_list)
# x的格式： [[第一条评论的切词分布，有就是1，没有就是0], [第二条评论的切词分布，有就是1，没有就是0], ...]
x = transfer.transform(comment_list).toarray()
print(x)

# 2.7看一下13条评论切词且删除停用词后剩下多少个词
print(transfer.get_feature_names_out())
print(len(transfer.get_feature_names_out()))        # 37个词，即13条评论切词且删除停用词后一共剩下多少个词

# 2.8因为就13条数据，我们把前10条当成训练集，后三条当测试集
x_train = x[:10]
y_train = y[:10]

x_test = x[10:]
y_test = y[10:]

# 3.特征工程，此处省略

# 4.模型训练
estimator = MultinomialNB() # 创建朴素贝叶斯模型对象
estimator.fit(x_train, y_train)
# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'模型预测结果：{y_pred}')

# 6.模型评估
print(f'准确率：{accuracy_score(y_test, y_pred)}')