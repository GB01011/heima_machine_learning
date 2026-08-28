# 导包
import matplotlib
matplotlib.use('TkAgg')   # 绘图之后单独新窗口显示图

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1.定义电力负荷模型类，配置日志，获取数据源
class PowerLoadModel:
    # 1.1初始化属性信息
    def __init__(self):
        # 1.2拼接日志文件名
        logfile_name = 'train_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # 1.3创建日志对象
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志
        self.logfile.info('开始创建电力负荷模型类的对象')
        # 1.4获取数据源
        self.data_source = data_preprocessing()

# 2.查看数据的整体分布情况
def ana_data(data):
    '''
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末是否有区别
    :param data:
    :return:
    '''
    # 0.为了防止会修改数据源，做一次复制
    ana_data = data.copy()

    # 1.查看数据整体情况
    ana_data.info()

    # 2.负荷整体的分布情况，直方图
    # 2.1创建画布
    fig = plt.figure(figsize=(12, 14))  # 原来40,80太夸张，改成合理尺寸宽12，高14
    fig.subplots_adjust(hspace=0.8)  # hspace：垂直子图之间的空隙，数值越大间隔越大，0~1之间
    # 2.2添加子图
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'], bins=20)      # 负荷，直方图，100个区间
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    # 3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 3.1新增1列，充当小时
    ana_data['hour'] = ana_data['time'].str[11:13]
    # 3.2根据小时分组，计算平均值
    hour_load_mean = ana_data.groupby('hour', as_index=False)['power_load'].mean()
    # print(hour_load_mean)   # [列1 hour， 列2 power_load 当前小时的平均负荷]
    # 3.3画出折线图
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xlabel('小时')

    # 4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    ana_data['month'] = ana_data['time'].str[5:7]
    month_load_mean = ana_data.groupby('month', as_index=False)['power_load'].mean()
    ax3 = fig.add_subplot(413)
    ax3.plot(month_load_mean['month'], month_load_mean['power_load'])    # x轴：月份  y轴：平均负荷
    ax3.set_title('各个月份的平均负荷趋势')
    ax3.set_xlabel('月份')

    # 5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末是否有区别
    ana_data['weekday'] = ana_data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    ana_data['is_holiday'] = ana_data['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)   # 0:周一到周五， 1：周末
    work_load_mean = ana_data[ana_data['is_holiday'] == 0].power_load.mean()
    holiday_load_mean = ana_data[ana_data['is_holiday'] == 1].power_load.mean()
    ax4 = fig.add_subplot(414)
    ax4.bar(['工作日', '周末'], [work_load_mean, holiday_load_mean])
    ax4.set_title('工作日与周末的平均负荷情况')
    ax4.set_xlabel('类别')



    plt.savefig('../data/fig/负荷整体的分布情况.png')
    plt.show()


# 4.测试
if __name__ == '__main__':
    # 4.1创建电力负荷模型类的对象
    pm = PowerLoadModel()
    # 4.2打印数据源
    # print(pm.data_source)

    # 4.3查看数据分布
    ana_data(pm.data_source)