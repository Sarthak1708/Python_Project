import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates

trans=pd.read_csv("Retail_Data_Transactions.csv")
print(trans)

response=pd.read_csv("Retail_Data_Response.csv")
print(response)

# ----------MERGING TWO TABLES------------
df=trans.merge(response,on='customer_id',how='left')
print(df)

# ---------FOR COUNT,MEAN,STD,MIN ETC--------
print(df.describe())

# ---------FOR MISSING VALUES------------
print(df.isnull().sum())

# --------DROPING NULL VALUES----------
df=df.dropna()
print(df)

# -------CHANGING DATATYPES-----------
df['trans_date']=pd.to_datetime(df['trans_date'])
df['response']=df['response'].astype('int64')
print(df)

print(df.dtypes)

# --------CHECK FOR OUTLIERS------------
# --------Z SCORE----------

# cal z score
# z_scores=np.abs(stats.zscore(df['trans_amount']))

#  set threshold
# threshold=3

# outliers=z_scores>threshold

# print(a[outliers])

# cal z score
# z_scores=np.abs(stats.zscore(df['response']))

# set threshold
# threshold=3

# outliers=z_scores>threshold

# print(a[outliers])

# ---------------BOX PLOT----------------

sns.boxplot(x=df['tran_amount'])
plt.show()

# creating new columns

df['month']=df['trans_date'].dt.month
print(df)

# which months have had the highest transaction amount

monthly_sales=df.groupby('month')['tran_amount'].sum()
monthly_sales=monthly_sales.sort_values(ascending=False).reset_index()
print(monthly_sales)

# customer having highest no of orders

customer_counts=df['customer_id'].value_counts().reset_index()
customer_counts.columns=['customer_id','count']
print(customer_counts)

# sort
top_5_cus=customer_counts.sort_values(by='count',ascending=False).head(5)
print(top_5_cus)

# ---------------BAR PLOT------------------

sns.barplot(x='customer_id',y='count',data=top_5_cus)
plt.show()

# customer having highest no of orders

customer_sales=df.groupby('customer_id')['tran_amount'].sum().reset_index()
# customer_counts.columns=['customer_id','count']
print(customer_sales)

# sort
top_5_sal=customer_sales.sort_values(by='tran_amount',ascending=False).head(5)
print(top_5_sal)

# ---------------BAR PLOT------------------

sns.barplot(x='customer_id',y='tran_amount',data=top_5_sal)
plt.show()

# ------------TIME SERIES ANALYSIS------------------

df['month_year']=df['trans_date'].dt.to_period('M')
monthly_sales=df.groupby('month_year')['tran_amount'].sum()

monthly_sales.index=monthly_sales.index.to_timestamp()

plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index,monthly_sales.values)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------COHORT SEGMENTATION--------------

# recency
recency=df.groupby('customer_id')['trans_date'].max()

# frequency
frequency=df.groupby('customer_id')['trans_date'].count()

# monetary
monetary=df.groupby('customer_id')['tran_amount'].sum()

# combine
rfm=pd.DataFrame({'recency':recency,'frequency':frequency,'monetary':monetary})

print(rfm)

# customer segmentation

def segment_customer(row):
    if row['recency'].year>=2012 and row['frequency']>=15 and row['monetary']>1000:
        return 'P0'
    elif(2011<=row['recency'].year<2012) and (10<row['frequency']<15)and (500<=row['monetary']<=1000):
        return 'P1'
    else:
        return 'P2'
    
rfm['segment']=rfm.apply(segment_customer,axis=1)

print(rfm)

# ---------CHURN ANALYSIS------------

# count the no of churned and active customer

churn_counts=df['response'].value_counts()

# plot
churn_counts.plot(kind='bar')
plt.show()

# -----------ANALYZING TOP CUSTOMER-----------

top_5_cus = monetary.sort_values(ascending=False).head(5).index

top_customer_df = df[df['customer_id'].isin(top_5_cus)]

top_customer_sales = top_customer_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)

top_customer_sales.plot(kind='line')
plt.show()

df.to_csv('MainData.csv')
rfm.to_csv('AddAnalysis.csv')
