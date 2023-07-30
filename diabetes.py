#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install scikit-learn')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[6]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, RocCurveDisplay


# In[7]:


from statistics import stdev
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# In[10]:


data = pd.read_csv(r"C:\Users\shiva\Documents\jupyter notebooks\diabetes_prediction_dataset.csv")


# In[11]:


samples, features = data.shape
print('Number Of Samples: ', samples)
print('Number Of Features: ', features)


# In[12]:


data.head(10)


# In[13]:


data.info()


# In[14]:


data.describe().T


# In[15]:


#UNIQUE VALUES


# In[16]:


d = []
u = []
t = []
for col in data:
    d.append(col)
    u.append(data[col].nunique())
    t.append(data[col].dtype)
pd.DataFrame({'column':d,'type': t ,'unique value' : u})


# In[17]:


labels = ['Female', 'Male', 'Other']
values = data['gender'].value_counts().values

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
sns.countplot(x=data['gender'], data=data)
plt.subplot(1, 2, 2)
plt.pie(values, labels=labels, autopct='%1.1f%%')

plt.savefig('FirstImage')
plt.show()


# In[18]:


labels = ['never', 'No Info', 'former', 'current', 'not current', 'ever']
values = data['smoking_history'].value_counts().values

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
sns.countplot(x=data['smoking_history'], data=data)
plt.subplot(1, 2, 2)
plt.pie(values, labels=labels, autopct='%1.1f%%')

plt.savefig('Image')
plt.show()


# In[19]:


numerical = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
i = 0

while i < 4:
  plt.figure(figsize=(20, 8))
  plt.subplot(1, 2, 1)
  sns.distplot(data[numerical[i]])
  i += 1
  if i == 4:
    break
  plt.subplot(1, 2, 2)
  sns.distplot(data[numerical[i]])
  i += 1
  plt.show()

plt.savefig('2')


# In[20]:


# PRE-PROCESSING AND CLEANINNG OF DATA


# In[21]:


data.isnull().sum()


# In[22]:


data = data.drop_duplicates()


# In[23]:


data.shape


# In[24]:


data['smoking_history'].replace({'never': 2, 'No Info': 3, 'current': 4, 'former': 5,
                                'not current': 6, 'ever': 7}, inplace=True)


# In[25]:


data['gender'].replace({'Male': 2, 'Female': 3, 'Other': 3}, inplace=True)


# In[26]:


data.head()


# In[27]:


#Exploring Categorical Features


# In[28]:


categorical_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history']


# In[29]:


fig, axes = plt.subplots(4,2, figsize=(20,30))
sns.set_style('darkgrid')
idx = 0
for col in categorical_columns:
    sns.countplot(data=data, y=col, palette='magma', orient='h',
                  ax=axes[idx][0]).set_title(f'Count of {col}', fontsize='20')
    for container in axes[idx][0].containers:
        axes[idx][0].bar_label(container)
    sns.countplot(data=data, y=col, palette='mako', orient='h',  hue='diabetes',
                  ax=axes[idx][1]).set_title(f'Count of {col} per Diabetes', fontsize='20')
    for container in axes[idx][1].containers:
        axes[idx][1].bar_label(container)
    idx +=1
plt.show()


# In[30]:


#Exploring Numerical Features


# In[31]:


numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# In[32]:


fig, axes = plt.subplots(4,2, figsize=(20,35))
sns.set_style('darkgrid')
idx = 0
for col in (numerical_columns):
    sns.kdeplot(data=data, x=col, palette='Greens',fill=True , hue='diabetes',
                ax=axes[idx][0]).set_title(f'Distribution of {col}', fontsize='16')
    sns.boxplot(data=data, x=col, palette='flare' , y='diabetes', orient='h',
                ax=axes[idx][1]).set_title(f'BoxPlot of {col}', fontsize='16')
    idx +=1
plt.show()


# In[33]:


#Correlation Between The Features


# In[34]:


fig=plt.gcf()
fig.set_size_inches(10,8)
plt.title('Correlation Between The Features')
a = sns.heatmap(data.corr(), annot=True, cmap='Pastel1', fmt='.2f', linewidths=0.2)
a.set_xticklabels(a.get_xticklabels(), rotation=60)
a.set_yticklabels(a.get_yticklabels())
plt.show()


# In[35]:


sns.set_palette(sns.color_palette("Paired", 8))
sns.pairplot(data, x_vars=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'], y_vars=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'], hue='diabetes',corner=True)
plt.show()


# In[36]:


#Data and Target Split


# In[37]:


target = data['diabetes']
data.drop('diabetes', axis=1, inplace=True)


# In[38]:


data.corrwith(target).plot.bar(
    figsize=(15, 10), title='Correlation with Diabetes',
    fontsize=15, rot=90, grid=True)
plt.savefig('5')
plt.show()


# In[39]:





# In[40]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




