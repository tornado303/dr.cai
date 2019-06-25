#!/usr/bin/env python
# coding: utf-8

# In[2]:


#导入所需要的包
import  pandas as pd
import numpy as np
import random
import sys
import time
import math
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#数据预处理
def dataCount(path):
    """
        整合读入的原始数据(traindata1017.csv)
        ：path:文件路径
        ：return:处理后的DataFrame对象
    """
    df = pd.read_csv(path,header=None,encoding='utf-8',engine='python') #读入数据
    data = df.T #数据进行转置
    list_count = []
    #遍历data里面的每一列数据选出满足条件的，并添加数据在原始数据中所在的行数标记(value)
    for i in range(len(data.columns)): 
        list_n = 'list_'+str(i) 
        list_n = []
        for close in data[i]:
            if close>1:
                y = [close,i]
                list_n.append(y)
            else:
                continue    
        list_count.extend(list_n)  
    df = pd.DataFrame(list_count,columns=['close','value'])#把所有的价格数据整合成一列构成一个新的DataFrame对象
    return df


# In[4]:


#数据去噪：
def dataDenoising(df,form='波谷'):
    """
      按每个波峰或波谷(form)，通过3-δ筛选噪点，并用噪点前后相邻非噪点数据的均值填充噪点
     :df:待去噪的数据
     :form：以何种形态去除噪声数据，默认'波谷',可选（'波谷','波峰'）
     :return: df(去噪后的DataFrame对象)    
    """
    #首先根据df中的'value'值进行波谷波峰划分
    df1 = df.groupby(by='value',as_index=False) #以'value'分组
    trough = [] #第一波谷列表用于存储数据
    crest = [1]*len(df[df['value']==0])#定义波峰列表用于存储数据
    trough_num = 1 #初始化波谷值
    crest_num = 1  #初始化波峰值
    for i in range(len(df1)-1):
        if i%2==0:
            arr = df[(df['value']==i) | (df['value']==i+1)]#取出一个波谷数据
            str1 = '波谷'+str(trough_num) #拼接波谷值,eg:'波谷1'，'波谷2'
            print(arr)
            trough_num+=1
            trough.extend([str1]*len(arr)) #把波谷的值存于波谷列表
        elif i%2==1:
            arr = df[(df['value']==i) | (df['value']==i+1)]#取出一个波峰数据
            str2 = '波峰'+str(crest_num) #拼接波峰值,eg:'波峰1'，'波峰2'
            crest_num+=1
            crest.extend([str2]*len(arr))#把波峰的值存于波谷列表
    crest.extend([1]*len(df[df['value']==4711]))

    df['波谷'] = trough #赋值给df
    df['波峰'] = crest  #赋值给df
    #按照单个波峰或波谷筛选噪声点
    df1 = df.groupby(by=form,as_index=False) #按照波谷或波峰对数据进行分组
    mean = df1['close'].mean() #求取每组的价格平均值
    mean = mean.rename(columns={'close':'mean'}) #重命名
    var = df1['close'].var() #求取每组的价格的方差
    var = var.rename(columns={'close':'var'}) #重命名
    df = pd.merge(left=df,right=mean,on=form,how='left') #以波谷或波峰字段为中间键，连接df和价格均值
    df = pd.merge(left=df,right=var,on= form,how='left') #以波谷或波峰字段为中间键，连接df和价格方差
    # print(df.head(10))
    df['std'] =  df['var'].map(lambda x:math.sqrt(x)) #通过价格方差，求取价格的标准差
    df = df.drop('var',axis=1) #去除价格的方差
    df1 = df[abs(df['close']-df['mean'])>3*df['std']]#3—δ方法筛选离群的点
    #噪声数据已找到，下面去噪
    num = 0
    print('数据去噪进度')
    for i in df1.index: #遍历噪声点的索引
        fla1 = True 
        fla2 = True
        num1 = 1
        num2 = 1
        while fla1: #寻找噪声点前一个相邻不是噪声点的数据
            if (i-num1) in list(df1.index):
                num1+=1
            else:
                x = df.loc[i-num1,'close']
                fla1 = False          
        while fla2:   #寻找噪声点后一个相邻不是噪声点的数据     
            if (i+num2)  in list(df1.index): 
                num2+=1
            else:
                y = df.loc[i+num2,'close']
                fla2 = False           
        df.loc[i,'close'] = (x+y)/2 #在噪声点的数据更换为相邻不是噪声数据的平均值
        d = 10
        done = int(num*100/(len( df1.index)*d))
        num+=1
        sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, '' * d,num*100/len( df1.index))) #噪声点数处理进度  
    df['close_sc'] = (df['close']-df['close'].min())/(df['close'].max()-df['close'].min()) #使用最大最小值便准化价格数据
    df = df[['close','value','波谷','波峰','close_sc']] #筛选字段
    print('去噪完成')
    return df   


# In[5]:


#特征提取
def dataFeatureExtraction(data):
    """
      分别提取/计算数据点个数、均值、中位数、标准差、方差、最大值、最小值、全距、波动率、偏度及峰度
      ：data:输入数据，待特征提取
      ：return:已提取的特征数据    
    """
    #定义需要提取特征的列表
    mean = [] #均值
    median = [] #中位数
    trend = [] #形态
    count = [] #数据数量
    var = [] #方差
    max1 = [] #最大值
    min1 = [] #最小值
    std = [] #标准差
    R = [] #全距
    CV = []#变异系数
    Skewness = [] #偏度
    kurtosis = [] #峰度
    wave_rate = [] #波动率
    for i in range(len( data.groupby(by=['value'],as_index=False).count())-1): #按照波峰或波谷提取数据特征
        list1 = list(data[data['value'] ==i]['close']) #取出第i组价格数据
        list2 = list(data[data['value']==i+1]['close']) #取出第i+1组价格数据
        list3 = list1+list2 # 两组价格数据组合在一起形成一个波谷或波峰
        arr = np.array(list3) #转化为numpy数组数据
        count.append(len(arr)) #统计波峰或波谷由多少个数据组成
        mean.append(np.mean(arr)) # 统计波峰或波谷这组价格数据的平均值
        var.append(np.var(arr)) #统计波峰或波谷这组价格数据的方差
        max1.append(np.max(arr)) #统计波峰或波谷这组价格数据的最大值
        min1.append(np.min(arr)) #统计波峰或波谷这组价格数据的最小值
        median.append(np.median(arr)) #统计波峰或波谷这组价格数据的中位数
        std.append(np.std(arr))   #统计波峰或波谷这组价格数据的标准差
        R.append(np.max(arr)-np.min(arr)) #统计波峰或波谷这组价格数据的全距
        CV.append(np.std(arr)/np.mean(arr)) #统计波峰或波谷这组价格数据的变异系数
        df_arr = pd.DataFrame() #定义一个pandas对象
        df_arr['close'] = arr  #把上面波峰或波谷数据转化成一个pandas对象数据
        Skewness.append(df_arr['close'].skew()) # 调用pandas.Series的api方法计算波峰或波谷的偏度
        kurtosis.append(df_arr['close'].kurt()) # 调用pandas.Series的api方法计算波峰或波谷的峰度
        list3 = list(data[data['value'] ==i]['close_sc']) # 取出第i组标准化后的价格数据
        list4 = list(data[data['value']==i+1]['close_sc']) # 取出第i+1组标准化后的价格数据
        list5 = list3+list4 # 两组数据组合成一个波峰或波谷
        arr = np.array(list5) #转化为numpy数组数据
        arr = np.sort(arr) #按照大小由小到大排序
        #拿这组数据90分位值减去10分位值作为这组数据的波动率
        x = len(arr)/10 
        y = 1+x
        j = 1+x*9
        i1 = int(y)
        i2 = y-i1
        j1 = int(j)
        j2 = j-j1
        q10 = arr[i1]+(arr[i1+1]-arr[i1])*i2 #10分位值
        q90 = arr[j1]+(arr[j1+1]-arr[j1])*j2 #90分位值
        wave_rate.append(q90-q10)
        if i%2==0: #判断这组数据是波峰或是波谷
            trend.append('波谷')
        else:
            trend.append('波峰')
    #把数据特征数据的列表统一放在同一个列表中        
    list2 = []
    list2.append(count)
    list2.append(mean)
    list2.append(var)
    list2.append(max1)
    list2.append(min1)
    list2.append(median)
    list2.append(std)
    list2.append(R)
    list2.append(CV)
    list2.append(Skewness)
    list2.append(kurtosis)
    list2.append(wave_rate)
    list2.append(trend)
    #由统计好的特征数据创建pandas文件
    data_1 = pd.DataFrame(list2,index=['count','mean','var','max','min','median','std','R','CV','Skewness','kurtosis','wave_rate','trend'])
    data= data_1.T #转置
    print('数据特征提取完成')
    return data


# In[6]:


def run(path,form='波谷'):
    '''
      数据特征提取运行函数
       ：path:文件地址;
       ：form:以造什么形态去除噪声数据，默认'波谷',可选（'波谷','波峰'）
       ：return:整合后的数据df1，去噪后的数据df2,及所需要的特征数据df3;    
    '''
    df1 = dataCount(path)#为整合后的价格数据
    df2 = dataDenoising(df1) #为去噪后的数据
    df3 = dataFeatureExtraction(df2)#为提取的数据特征
    return df1,df2,df3


# In[7]:


def comparisonDenoising(df1,df2):
    '''
        通过图形比对数据去噪前和去噪后的差异；
        df1:去噪前的数据；
        df2:去噪后的数据
        return:None
    '''
    plt.figure(figsize=(40, 10))
    plt.scatter(df1.index,df1['close'], s=5,c='b',label='去噪前数据')
    plt.scatter(df2.index,df2['close'], s =5,c='r',label='去噪后数据')
    plt.ylabel('价格')
    plt.xlabel('时间')
    plt.title('波谷')
    plt.show()


# In[11]:


# 传入文件地址进行数据特征提取
path = r'traindata1017.csv'
df1,df2,df3 = run(path=path)


# In[9]:


# 查看去噪后价格数据和去噪前价格数据对比图
#comparison_denoising(df1,df2)


# In[10]:


df3.to_csv('feature_5.csv',sep=',',index=False)
#df2.to_csv('去噪后数据.csv',sep=',',index=False)

