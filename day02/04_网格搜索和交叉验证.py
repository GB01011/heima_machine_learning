"""
案例：演示网格搜索和交叉验证

交叉验证解释：
    原理：
        把数据分成n份，例如分成4份：————也称4折交叉验证
        第1次：把第1份数据作为 验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率 ———— 准确率1
        第2次：把第2份数据作为 验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率 ———— 准确率2
        第3次：把第3份数据作为 验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率 ———— 准确率3
        第4次：把第4份数据作为 验证集（测试集），其他作为训练集，训练模型，模型预测，获取：准确率 ———— 准确率4

        假设第4次最好（准确率最高），则：用全部数据（训练集 + 测试集）训练模型，再次用（第4次的）测试集对模型测试
    目的：
        为了让模型的最终验证结果更准确

网格搜索：
    目的/作用：
        寻找最优超参数
    原理：
        接收超参可能出现的值，然后针对于超参的每个值进行交叉验证，获取到最优超参组合
    超参数：
        需要用户手动录入的数据，不同的超参（组合）可能会影响模型的最终评测结果

即：
    网格搜索+交叉验证，本质上指的是GridSearchCV这个API，它会帮我们寻找最优超参
"""

# 导入工具包
from sklearn.datasets import load_iris                              # 加载鸢尾花测试集
from sklearn.model_selection import train_test_split,GridSearchCV   # 分割训练集和测试集,寻找最优超参（网格搜索+交叉验证）
from sklearn.preprocessing import StandardScaler                    # 数据标准化
from sklearn.neighbors import KNeighborsClassifier                  # KNN算法分类对象
from sklearn.metrics import accuracy_score                          # 模型评估，计算模型预测的准确率

# 1.加载鸢尾花数据集
iris_data = load_iris()

# 2.数据预处理，这里是：切分训练集和测试集，比例：8：2
# 参1:特征数据   参2:标签数据   参3:测试集的比例   参4:随机种子（种子一致，每次生成的随机数据集都是固定的）
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,random_state=22)

# 3.特征工程 ———— 特征预处理 ———— 标准化
# 3.1创建标准化对象
transfer = StandardScaler()
# 3.2对训练集和测试集的特征数据进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
# 4.1创建KNN分类对象
estimator = KNeighborsClassifier()
# 4.2定义字典,记录超参可能出现的情况(值)
param_dict = {'n_neighbors':[i for i in range(1,11)]}       # i的值:1~10
# 4.3创建GridSearchCV对象,使用网格搜索+交叉验证方式寻找最优超参
# 参1:要计算最优超参的模型对象
# 参2:该模型超参可能出现的值
# 参3:交叉验证的折数,这里的4折表示:每个超参组合都会进行4次交叉验证,这里共计是 4 * 10 = 40次
# 返回值estimator:处理后的模型对象
estimator = GridSearchCV(estimator,param_dict,cv=4)
# 4.4具体的模型训练动作
estimator.fit(x_train,y_train)
# 4.5打印最优超参组合
print(f'最优评分: {estimator.best_score_}')
print(f'最优超参组合:{estimator.best_params_}')
print(f'最优的估计器对象:{estimator.best_estimator_}')
print(f'具体的交叉验证结果:{estimator.cv_results_}')

# 5.模型评估
# 5.1获取最优超参的模型对象
estimator = estimator.best_estimator_       # 获取最优的模型对象
# 也可以像下面这样直接把最优超参传进来
estimator = KNeighborsClassifier(n_neighbors=3)
# 5.2模型训练
estimator.fit(x_train,y_train)
# 5.3模型预测
y_pre = estimator.predict(x_test)
# 5.4模型评估
# 参1:测试集      参2:预测集
print(f'准确率: {accuracy_score(y_test,y_pre)}')