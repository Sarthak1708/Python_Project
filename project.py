import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------TO READ THE CSV FILE--------------------------

df=pd.read_csv("Expanded_data_with_more_features.csv.zip")
# print(df.head())

# --------------TO CHECK COLUMNS INDEX--------------------
# print(df.columns)

# print(df.describe())

# df.info()

# df.isnull().sum()

# ------------DROPING UNNECESSARY COLUMNS--------------
# df = df.drop("Unnamed: 0", axis=1)
# print(df.head())

#----------------REPLACE THE DATE WITH TIME----------------- 

# df["WklyStudyHours"]=df["WklyStudyHours"].str.replace("05-Oct","5-10")
# print(df.head())

# pd.set_option('display.max_columns', None)
# print(df)

#-------------------PLOT THE BAR GRAPH OF GENDER------------------- 

# plt.figure(figsize=(5,5))
# ax=sns.countplot(data=df,x="Gender")
# ax.bar_label(ax.containers[0])
# plt.title("Gender distribution")
# plt.show()

# thus from the above data we analyse that the no. of females are more than males 

# gb=df.groupby("ParentEduc").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
# print(gb)

#------------------TO MAKE HEATMAP-------------------- 

# plt.figure(figsize=(4,4))
# sns.heatmap(gb,annot=True)
# plt.title("Relationship b/w parent edu and std score")
# plt.show()

#from above we analyse that education of parents have a good impact on their students 

# gb1=df.groupby("ParentMaritalStatus").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
# print(gb1)

# plt.figure(figsize=(4,4))
# sns.heatmap(gb1,annot=True)
# plt.title("Relationship b/w parent marital status and std score")
# plt.show()

# here we have concluded that there is no or negligible impact on the 
# student's score due to their parents marital status


#----------------------PLOTTING BOX PLOT OF DIFFERENT SCORES------------------- 

# plt.boxplot(df["MathScore"])
# plt.xlabel('MathScore')
# plt.show()

# sns.boxplot(data=df,x="ReadingScore",orient='h')
# plt.show()

# sns.boxplot(data=df,x="WritingScore",orient='h')
# plt.show()    

# ---------------TO FIND UNIQUE GROUP----------------------

# print(df["EthnicGroup"].unique())

# ----------------------------TO PLOT PIE CHART OF ETHNIC GRP---------------------

# groupA=df.loc[(df['EthnicGroup'] == "group A")].count()
# groupB=df.loc[(df['EthnicGroup'] == "group B")].count()
# groupC=df.loc[(df['EthnicGroup'] == "group C")].count()
# groupD=df.loc[(df['EthnicGroup'] == "group D")].count()
# groupE=df.loc[(df['EthnicGroup'] == "group E")].count()

# l=["group A","group B","group C","group D","group E"]
# mlist=[groupA["EthnicGroup"],groupB["EthnicGroup"],groupC["EthnicGroup"],groupD["EthnicGroup"],groupE["EthnicGroup"],]

# print(mlist)
# plt.pie(mlist,labels=l,autopct="%1.2f%%")
# plt.title("Distribution of Ethnic group")
# plt.show()

# ---------------------TO SHOW COUNT OF ETHNIC GRP------------------------
# ax=sns.countplot(data=df,x='EthnicGroup')
# ax.bar_label(ax.containers[0])
# plt.show()