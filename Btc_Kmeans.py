import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import functools
import pickle
import itertools
import os

# %matplotlib inline


pd.options.mode.chained_assignment = None

columns_list = ['mean', 'var', 'std', 'wave_rate']#取所需字段


def main(csv_path='feature_4.csv', base_dir = 'fail_csv1'):
    df = pd.read_csv(csv_path)#读取CSV文件
    df1 = df.groupby(df['trend'])#按trend对数据进行分组
    _peak, _valley = map(lambda x: x[1], list(df1))#分成波峰和波谷


    # 对波峰数据进行分析
    dict_peak = {}
    for numb_columns in [2, 3, 4]:
        d = {}
        # 对特征进行排列组合
        for columns in itertools.combinations(columns_list, numb_columns):
            print('')
            # 打印训练的记录
            print("{}当前状态：numb_columns：{} columns：{}".format('peak', numb_columns, list(columns)))
            # 训练单个组合的数据
            split_result = split_data_with_possibility_and_kmenas(_peak, list(columns), base_dir, types='peak')
            d['+'.join(columns)] = split_result# 将组合名作为关键字储存
        dict_peak[str(numb_columns)] = d# 2/3/4作为键值存储

    # 对波谷数据进行分析
    dict_valley = {}
    for numb_columns in [2, 3, 4]:
        d = {}
        for columns in itertools.combinations(columns_list, numb_columns):
            print('')
            # 打印训练的记录
            print("{}当前状态：numb_columns：{} columns：{}".format('valley', numb_columns, list(columns)))
            # 训练单个组合的数据
            split_result = split_data_with_possibility_and_kmenas(_valley, list(columns), base_dir, types='valley')
            d['+'.join(columns)] = split_result# 将组合名作为关键字储存
        dict_valley[str(numb_columns)] = d# 2/3/4作为键值存储

    result = {}
    result['peak'] = dict_peak#存所有的波峰训练结果
    result['valley'] = dict_valley#存所有波谷训练结果

    with open('result_with_winrate1.pkl', 'wb') as f:#将训练完后的数据存成.pkl文件
        pickle.dump(result, f)


# 对数据进行归一化
def zscore_data(f1):
    return (f1 - f1.mean(axis=0)) / (f1.std(axis=0))


def split_data_with_possibility_and_kmenas(df_data, columns=None, base_dir=None, types=None):
    """
    :param df_data:聚类所需字段
    :param columns: 要聚类的列，e.g., ['mean', 'std']
    :param base_dir:fail记录存放路径
    :param types:波峰 or 波谷
    :return: 字典dict{"k2": [[1,2,3][][]], "k3": [[1,2,3][][]]}
    """
    print("-" * 10)

    df1 = df_data[columns + ['R']]#聚类所需字段加上字段R的数据

    k2_list, k3_list = [], []
    #对fail的数据存成csv文件进行命名
    k2_save_fail_path = os.path.join(base_dir, types + '：' + '+'.join(columns) + '-k2-')
    k3_save_fail_path = os.path.join(base_dir, types + '：' + '+'.join(columns) + '-k3-')

    for test_size in [0.5, 0.33, 0.2]:#对训练数据和测试数据进行分割
        train_data, test_data = train_test_split(df1, test_size=test_size, random_state=0)
        #print(test_data)
        #k=2时，单条数据训练测试并存储fail的记录
        k2 = kmeans_one_data(train_data, test_data, n_clusters=2, columns=columns,
                             save_fail_path=k2_save_fail_path + str(test_size) + '.csv')
        # k=3时，单条数据训练测试并存储fail的记录
        k3 = kmeans_one_data(train_data, test_data, n_clusters=3, columns=columns,
                             save_fail_path=k3_save_fail_path + str(test_size) + '.csv')
        print("test_size:", test_size, k2, k3)
        k2_list.append(list(k2))
        k3_list.append(list(k3))
    # print("=" * 10 + "\n", k2_list, "\n", k3_list)
    d = {}#定义个字典将k=2和k=3的数据存入字典中
    d['k2'] = k2_list
    d['k3'] = k3_list

    return d


def kmeans_one_data(df_train, df_test, n_clusters=2, columns=None, save_fail_path=None):    
    """    
    :df_train:训练数据
    :df_test:测试数据
    :n_cluster:类cluster数
    :columns:要聚类的列，e.g., ['mean', 'std']
    :return:(测试集pass的记录个数，测试记录总数，胜率，盈率，fail list[456, 454,746,...])
    """

    # first kmeans
    df_train['result'] = 'train'#对训练数据加上train标签
    df_test['result'] = 'test'#对测试数据加上test标签
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, random_state=0, n_jobs=-1) #聚类
    columns_index = len(columns) + 1#新加test列数据的位置	

    caculate = 0
    for x in range(len(df_test.index)):		
    #for x in range(len(df_test.index)):
        #将test中的数据一条一条进行测试
        print(len(df_test.index))
        tmp_train_data = pd.merge(df_train, df_test[x:x + 1], how='outer')
        kmeans.fit(zscore_data(tmp_train_data[columns]))#对数据进行拟合
        tmp_train_data['label'] = kmeans.labels_#将聚类后的数据新加一列label分群	
        

        #找出每个群中的Rmax
        center = tmp_train_data[:-1].groupby('label').max().reset_index()	
        
        if n_clusters == 2:
            R0_max, R1_max = center['R']#当k=2时的两个R最大值
        else:
            R0_max, R1_max, R2_max = center['R']#当k=3时的三个R最大值        
        last_data = tmp_train_data[-1:]#新加入的一条test记录		

        if float(last_data['label']) == 0:#判断新加的test记录属于哪个群            
            if float(last_data['R']) < R0_max:#如果新加的test记录中的R值小于该群R最大值
                df_test.iloc[x, columns_index] = 'pass'#将通过的记录进行标记为pass
                df_train = tmp_train_data#将通过的记录加入训练集形成新的训练集
            else:#如果大于该群R最大值则标记为fail                
                df_test.iloc[x, columns_index] = 'fail'

        elif float(last_data['label']) == 1:#判断新加的test记录属于哪个群            
            if float(last_data['R']) < R1_max:#如果新加的test记录中的R值小于该群R最大值
                df_test.iloc[x, columns_index] = 'pass'#将通过的记录进行标记为pass
                df_train = tmp_train_data#将通过的记录加入训练集形成新的训练集
            else:#如果大于该群R最大值则标记为fail                
                df_test.iloc[x, columns_index] = 'fail'

        else:
            if float(last_data['R']) < R2_max:#如果新加的test记录中的R值小于该群R最大值
                df_test.iloc[x, columns_index] = 'pass'#如果新加的test记录中的R值小于该群R最大值
                df_train = tmp_train_data#将通过的记录加入训练集形成新的训练集
            else:#如果大于该群R最大值则标记为fail
                df_test.iloc[x, columns_index] = 'fail'
    #将fail的记录取出
    df_fail = df_test[df_test['result'] == 'fail']
    
    #将fail的记录保存成csv文件
    df_fail.to_csv(save_fail_path)
    print('fail:',list(df_fail.index))

    sum_pass_R = df_test[df_test['result'] == 'pass']['R'].sum()#统计所有pass记录R的总和
    sum_test_R = df_test['R'].sum()#统计所有test记录R的总和

    numb_pass = sum(df_test['result'] == 'pass')#统计pass记录的个数
    numb_test_all = len(df_test)#统计所有test记录的个数

    win_rate =  float(sum_pass_R) /  float(sum_test_R)#计算盈率
    #测试集pass的个数, test总数， 胜率, 盈率, fail list[456, 454,746,...]
    return numb_pass, numb_test_all, numb_pass / numb_test_all, win_rate, list(df_fail.index)


if __name__ == "__main__":
    main()
