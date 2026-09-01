# 导包
import matplotlib
matplotlib.use('TkAgg')   # 绘图之后单独新窗口显示图

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
import matplotlib.ticker as mick
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1.配置电力负荷预测类
class PowerLoadPredict(object):
    def __init__(self, file_path):
        # 配置日志记录
        logfile_name = 'predict_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logger = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(file_path)
        # 历史数据转为字典, key:时间, value:负荷,目的是避免频繁操作dataframe, 提高效率, 实际开发场景可以使用redis进行缓存
        # 格式为：{'2015-07-31 00:00:00': 801.59, '2015-07-31 01:00:00': 764.79,...}
        self.time_load_dict = self.data_source.set_index('time').power_load.to_dict()


# 2.预测数据解析特征，保持与模型训练时的特征列名一致
def pred_feature_extract(data_dict, time, logger):
    '''
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征
    2.解析时间窗口特征
    3.解析昨日同时刻特征
    :param data_dict: 历史数据，字典格式，key：时间， value：负荷
    :param time: 预测时间，字符串类型，格式为：2024-12-20 09:00:00
    :param logger: 日志对象
    :return:
    '''
    logger.info(f'========解析预测时间为：{time}所对应的特征================')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']

    # 1.解析时间特征，即：time字段（预测时间）对应的那条数据样本
    # 1.1截取要预测的time字段的小时信息
    pre_hour = time[11:13]  # 例如：'2015-08-30 04:00:00' ———— '04'
    hour_list = []
    for i in range(24):
        if pre_hour == feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    # print(hour_list)

    # 1.2截取要预测的time字段的月份信息
    pre_month = time[5:7]   # 例如：'2015-08-30 04:00:00' ———— '08'
    month_list = []
    for i in range(24, 36):
        if pre_month == feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    print(month_list)


    # 2.解析时间窗口特征
    # 2.1前1个小时特征
    last_1h_time = (pd.to_datetime(time) - pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 500) # 获取前一个小时对应的负荷，若没有就默认为500

    # 2.2前2个小时特征
    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 500) # 获取前两个小时对应的负荷，若没有就默认为500

    # 2.3前3个小时特征
    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 500) # 获取前三个小时对应的负荷，若没有就默认为500

    # 3.解析昨天同时刻特征
    yesterday_time = (pd.to_datetime(time) - pd.to_timedelta('1D')).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load = data_dict.get(yesterday_time, 500)

    # 4.拼接特征数据
    feature_data = hour_list + month_list + [last_1h_load, last_2h_load, last_3h_load, yesterday_load]

    # 5.转成df对象并返回
    feature_df = pd.DataFrame([feature_data], columns=feature_names)
    # print(feature_df)
    return feature_df

# 3.绘制时间与预测负荷折线图，时间与真实负荷折线图
def prediction_plot(data):
    '''
    绘制时间与预测负荷折线图，时间与真实负荷折线图，展示预测效果
    :param data: 数据一共有三列：时间、真实值、预测值
    :return:
    '''
    # 1.创建画布
    fig = plt.figure(figsize=(40, 26))
    # 2.创建子图
    ax = fig.add_subplot()
    # 3.绘制时间与预测负荷折线图
    ax.plot(data['预测时间'], data['真实负荷'], color='blue', label='真实负荷')
    ax.plot(data['预测时间'], data['预测负荷'], color='red', label='预测负荷')
    # 4.添加标题
    ax.set_title('真实负荷和预测负荷关系图', fontsize=30)
    ax.set_xlabel('时间')
    ax.set_ylabel('负荷')
    # 5.添加网格
    ax.grid(True, linestyle='--', alpha=0.5)
    # 6.添加图例
    ax.legend(loc='best', fontsize=20)
    # 7.设置刻度间隔
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(base=50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    # 8.保存图例
    plt.savefig('../data/fig/真实负荷和预测负荷关系图.png')
    # 9.展示图片
    plt.show()



# 4.测试
if __name__ == '__main__':
    # 4.1创建电力负荷预测类对象
    pp = PowerLoadPredict('../data/test.csv')
    # print(pp.data_source)
    # print(pp.time_load_dict)
    # 4.2加载模型对象
    estimator = joblib.load('../model/xgb_20260831.pkl')

    # 4.3确定要预测的时间段（2015-08-01 00:00:00及以后的时间）
    pre_times = pp.data_source['time'][pp.data_source['time'] >= '2015-08-01 00:00:00']
    # print(pre_times)

    # 定义evaluate_list列表，用于保存预测结果，方便后续进行预测结果评价
    evaluate_list = []

    # 4.4为了模拟实际场景的预测，把要预测的时间以及以后的负荷都掩盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
    for pre_time in pre_times:      # pre_time:'2015-08-01 00:00:00','2015-08-01 01:00:00',...
        # print(f'正在预测 {pre_time} 时间的负荷...')
        pp.logger.info(f'正在预测 {pre_time} 时间的负荷...')
        time_load_dict_masked = {k: v for k, v in pp.time_load_dict.items() if k <= pre_time}
        # print(time_load_dict_masked)

        # 4.5预测负荷
        # 4.5.1解析特征（定义解析特征方法）
        feature_df = pred_feature_extract(time_load_dict_masked, pre_time, pp.logger)

        # 4.5.2利用加载的模型预测
        y_pred = estimator.predict(feature_df)
        print(f'预测值为：{y_pred}')

        # 4.6保持预测时间对应的真实负荷
        true_value = pp.time_load_dict.get(pre_time, 500)


        # 4.7结果保持到evaluate_list， 三个元素分别是预测时间、真实负荷、预测负荷，方便后续进行预测结果评价
        evaluate_list.append([pre_time, true_value, y_pred[0]])
        pp.logger.info(f'预测时间：{pre_time}, 真实负荷：{true_value}, 预测负荷：{y_pred[0]}')


    # 4.8预测结束后, evaluate_list转为DataFrame
    evaluate_df = pd.DataFrame(evaluate_list, columns=['预测时间', '真实负荷', '预测负荷'])
    # print(evaluate_df)

    # 5.预测结果评价
    # 5.1计算预测结果与真实结果的MAE
    print(f'平均绝对误差：{mean_absolute_error(evaluate_df['真实负荷'], evaluate_df['预测负荷'])}')

    # 5.2绘制折线图（预测时间-真实负荷折线图，预测时间-预测负荷折线图），查看预测效果
    prediction_plot(evaluate_df)