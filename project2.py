import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Diwali Sales Data.csv",encoding='unicode_escape')
# print(df.head())

# df.info()

# ---------------------TO REMOVE UNNECESSARY COLUMN--------------------
df.drop(['Status','unnamed1'],axis=1,inplace=True)
print(df.head())

# -------------------TO FIND HOW MANY NULL ELEMENTS ARE THERE----------------
df1=df.isnull().sum()
print(df1)

# ------------------TO DROP NULL VALUES------------------------
df.dropna(inplace=True)
print(df.head())

null_counts_after_drop = df.isnull().sum()
print(null_counts_after_drop)

# ------------------TO CHANGE DATA TYPE--------------------
df['Amount']=df['Amount'].astype('int')
print(df.dtypes)

# ------------------DISPLAY HEADING OF COLUMNS-----------------
# print(df.columns)

# -----------------RETURN DESCRIPTION OF DATA(i.e MEAN,COUNT....)----------------
# print(df.describe())

# -------------------TO SHOW GENDER IN COUNT PLOT-------------------------
# ax=sns.countplot(x='Gender',data=df)
# for bar in ax.containers:
#     ax.bar_label(bar)
#     plt.show()

# -------USING GROUPBY TO GROUP GENDER AND AMOUNT TO SHOW THE PURSHASING POWER ------
# sales_gen=df.groupby(['Gender'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)
# sns.barplot(x='Gender',y='Amount',data=sales_gen)
# plt.show()
# from the graph we can see that most of the buyers are females and even the 
# purchasing power of females are greater than men 

# ------------------COUNT PLOT OF AGE GROUP AND GENDER---------------
# ax=sns.countplot(data=df,x='Age Group',hue='Gender')
# for bars in ax.containers:
#     ax.bar_label(bars)
# plt.show()

# ----------------------TOTAL AMOUNT VS AGE GROUP-------------------
# sales_age=df.groupby(['Age Group'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)
# sns.barplot(x='Age Group',y='Amount',data=sales_age)
# plt.show()
# from the graph we can see that the most of the buyers are from the age group b/w 26-35 females

# -----------------TOTAL NO OF ORDERS FROM TOP 10 STATES----------------
# sales_state=df.groupby(['State'],as_index=False)['Orders'].sum().sort_values(by='Orders',ascending=False).head(10)
# sns.barplot(x='State',y='Orders',data=sales_state)
# sns.set(rc={'figure.figsize':(20,10)})
# plt.show()

# -----------------TOTAL NO OF SALES FROM TOP 10 STATES----------------
# sales_amount=df.groupby(['State'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)
# sns.barplot(x='State',y='Amount',data=sales_amount)
# sns.set(rc={'figure.figsize':(20,10)})
# plt.show()
# from the above graph most orders are from UP,MH,karnataka but total sales is from UP,karnataka and MH

# ----------------GRAPH TO SHOW COUNT OF MARITAL STATUS-------------------
# ax=sns.countplot(x='Marital_Status',data=df)
# for bars in ax.containers:
#     ax.bar_label(bars)
# sns.set(rc={'figure.figsize':(7,5)})
# plt.show()

# --------------------TO SHOW AMOUNT OF SHOPPING DONE BY WHICH GENDER--------------
# sales_marstat=df.groupby(['Marital_Status','Gender'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)
# sns.barplot(x='Marital_Status',y='Amount',data=sales_marstat,hue='Gender')
# sns.set(rc={'figure.figsize':(15,10)})
# plt.show()
# from the above graph we can see that most of the buyers are maaried(women)and they have high purchasing power

# --------------GRAPH TO SHOW PRODUCT CATEGORY--------------------
# ax=sns.countplot(x='Product_Category',data=df)
# for bars in ax.containers:
#     ax.bar_label(bars)
# sns.set(rc={'figure.figsize':(20,10)})
# plt.show()

# ----------------------GRAPH OF PRODUCT CATEGORY AND AMOUNT-----------------------
# sales_procat=df.groupby(['Product_Category'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)
# sns.barplot(x='Product_Category',y='Amount',data=sales_procat)
# sns.set(rc={'figure.figsize':(20,5)})
# plt.show()
# from the above we can see that most of the sold products are from food,clothing and electronic category

# sales_procid=df.groupby(['Product_ID'],as_index=False)['Orders'].sum().sort_values(by='Orders',ascending=False).head(10)
# sns.barplot(x='Product_ID',y='Orders',data=sales_procid)
# sns.set(rc={'figure.figsize':(20,5)})
# plt.show()