#!/usr/bin/env python
# coding: utf-8

# # 频繁模式与关联规则挖掘

# ## 1. 原始数据概要

# 数据集给出了 2011-2016 年奥克兰的犯罪情况记录。以下是每列数据的统计情况：
# - Agency：机构名称，所有文件除了缺失值，该属性只有唯一值 OP，资料及讨论区并没有给出具体含义，应该是指警方来源；
# - Create Time：案件创建时间；
# - Location：案件发生的地点；
# - Area ID：案件发生区域代码；
# - Beat：案件发生的巡逻区；
# - Priority：案件优先级，所给所有数据集中：
#     - 0：Emergency call which requires immediate response and there is reason to believe that an immediate threat to life exists.
#     - 1：Emergency call which requires immediate response and there exists an immediate and substantial risk of major property loss or damage.
#     - 2：相较于 0 和 1 的优先程度较低；
# - Incident Type ID：案件类型 ID；
# - Incident Type Description：案件类型；
# - Event Number：案件标号，唯一值；
# - Closed Time：案件结束时间。

# ## 2. 项选择与数据集转换

# ### 不选择
# Agency：全部数据的 Agency 项值都为 OP，为无意义项；
# Event Number：案件标号，每个数据项是唯一值，为无意义项。
# 
# ### 概念重叠、包含，选择其一
# Incident Type ID 与 Incident Type Descreiption 的值相互对应，属于同一内容；
# Area，Beat，Location 均为案件发生的地点，概念重叠，故仅选择 Beat 构成新的数据集。
# 
# ### 提取隐含信息
# 由 Create Time 案件创建时间和 Closed Time 案件结束时间可以获取到案件的持续时间，生成新的项 Duration 加入新的数据集。
# 
# ### 概念收缩、分层
# 将 Start Time 维度收缩到 Month，将所述时间用对应的月份表示，Closed Time 同样。
# 将新增的 Duration 按时间进行分层（单位：Min）
# - 1：对应案件解决时间在 0-10 Min；
# - 2：对应案件解决时间在 10-30 Min；
# - 3：对应案件解决时间在 30-60 Min；
# - 4：对应案件解决时间在 1-3 Hour；
# - 5：对应案件解决时间在 3-6 Hour；
# - 6：对应案件解决时间在 6-12 Hour；
# - 7：对应案件解决时间在 12-24 Hour；
# - 8：对应案件解决时间在 > 1 Day。
# 
# 最终处理后的数据由6列构成，分别为 Create Time、Beat、Priority、Incident Type Description、Closed Time 和 Duration。

# In[20]:


import numpy as np
import pandas as pd
def load_data(filename):
    df = pd.read_csv(filename)
    return df.dropna()


# In[21]:


# 2016年为例
df = load_data('records-for-2016.csv')
df.head()


# In[22]:


import datetime
def str_to_datetime(s):
    date,time = s.split('T')
    date = date.split('-')
    time = time[:-4].split(':')
    date = [int(x) for x in date]
    time = [int(x) for x in time]
    return datetime.datetime(date[0],date[1],date[2],time[0],time[1],time[2])

def hierarchy(time):
    conditions = lambda x: {
        x < 10: 1, 
        10 <= x < 30: 2,
        30 <= x < 60: 3,
        60 <= x < 3*60: 4,
        3*60 <= x < 6*60: 5,
        6*60 <= x < 12*60: 6,
        12*60 <= x < 24*60: 7,
        x >= 24*60: 8,
    }
    return conditions(time)[True]

    
def time_interval(col1,col2):
    start = col1.values;
    end = col2.values;
    
    ans = []
    for s,e in zip(start,end):
        time = int((str_to_datetime(e)-str_to_datetime(s)).seconds/60)
        ans.append(hierarchy(time))
    return ans

def to_month(col):
    ans = []
    for i in col.values:
        date = str_to_datetime(i)
        ans.append(date.month)
    return ans

def process_data(df, filename):
    df = load_data(filename)
    # 不选择
    df = df.drop(['Agency', 'Location', 'Area Id', 'Incident Type Id', 'Event Number'], axis = 1)
    # 提取隐含信息并分层
    df['Duration'] = time_interval(df['Create Time'], df['Closed Time'])
    # 时间信息压缩
    df['Create Time'] = to_month(df['Create Time'])
    df['Closed Time'] = to_month(df['Closed Time'])
    df.to_csv(filename, index=False)
    df = load_data(filename)
    return df


# In[24]:


# 处理数据
df = process_data(df, 'records-for-2016.csv')
df.head()


# 使用 Python 的 mlxtend 库进行数据挖掘，需要将数据处理成其定义模型能够读取的形式。

# In[25]:


from mlxtend.preprocessing import TransactionEncoder

def deal(data):
    return data.to_list()


def add_prefix(df, filename):
    df['Create Time'] = ['cre' + str(x) for x in df['Create Time'].values]
    df['Closed Time'] = ['cls' + str(x) for x in df['Closed Time'].values]
    df['Priority'] = ['p' + str(x) for x in df['Priority'].values] 
    df['Duration'] = ['d' + str(x) for x in df['Duration'].values]
    df.to_csv(filename, index=False)
    df = load_data(filename)
    return df


# In[26]:


# 给create time、priority、closed time、duration列加上前缀进行区分
df = add_prefix(df, 'records-for-2011.csv')
df.head()


# In[27]:


# 转化成列表
df_arr = df.apply(deal,axis=1).tolist()


# In[28]:


# TransactionEncoder类似于独热编码，每个值转换为一个唯一的bool值
te = TransactionEncoder()
df_tf = te.fit_transform(df_arr)
df = pd.DataFrame(df_tf,columns=te.columns_)
df.head()


# ## 3. 频繁模式计算

# In[29]:


from mlxtend.frequent_patterns import apriori

frequent_items = apriori(df, min_support=0.05, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)
frequent_items.head(20)


# ## 4. 关联规则导出与评价

# 使用 Lift、Allconf、Cosine、Jaccard、Maxconf 以及 Kulczynski 规则评价关联规则。

# In[18]:


import math

def metrics(r,f):
    ans = []
    for i in range(r.shape[0]):
        item = r.iloc[i]
        ans.append(f(item))
    return ans

def allconf(item):
    return item.support/max(item['antecedent support'],item['consequent support'])

def cosine(item):
    return item.support/math.sqrt(item['antecedent support']*item['consequent support'])

def Jaccard(item):
    return item.support/(item['antecedent support']+item['consequent support']-item.support)

def maxconf(item):
    return max(item.support/item['antecedent support'],item.support/item['consequent support'])

def Kulczynski(item):
    return 0.5*(item.support/item['antecedent support']+item.support/item['consequent support'])


# In[30]:


from mlxtend.frequent_patterns import association_rules	

def gen_rules(frequent_items):
    rules =  association_rules(frequent_items, metric='lift')
    rules = rules.sort_values(by=['lift'], ascending=False).reset_index(drop=True)
    rules = rules.drop(['leverage','conviction'],axis = 1)
    rules['cosine'] = metrics(rules,cosine)
    rules['Jaccard'] = metrics(rules,Jaccard)
    rules['Allconf'] = metrics(rules,allconf)
    rules['Maxconf'] = metrics(rules,maxconf)
    rules['Kulczynski'] = metrics(rules,Kulczynski)
    return rules

rules = gen_rules(frequent_items)

rules.head(20)


# ## 5. 挖掘结果分析与可视化

# ### 挖掘结果分析

# 从表种看出，所给的这些规则的 lift 都是远大于 1 的，说明项之间的正相关：
# - 规则说明在1、2、3、42月发生的优先度为 2（最低优先度，程度轻微，如财产安全受到侵害）的案件通常都会在当月得到解决。即程度轻微的简单案件一般都能在当月解决；
# - 与2011年进行比较，可以一定程度上说明治理能力有了一定幅度的提升。

# ### 关联规则可视化

# 横坐标表示规则的支持度，纵坐标表示规则的置信度，其颜色的深浅代表了 Lift 的大小（颜色越深，Lift 越大）。

# In[31]:


import  matplotlib.pyplot as plt

def draw_rule(rules):
    plt.xlabel('support')
    plt.ylabel('confidence')
    for i in range(rules.shape[0]):
        plt.scatter(rules.support[i],rules.confidence[i],s=20,c='b',alpha=(rules.lift.iloc[i])/(rules.lift.iloc[0])*0.8/(rules.lift.iloc[0]-rules.lift.iloc[-1])+0.3)


# In[32]:


draw_rule(rules)

