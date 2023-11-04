#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


loan_train = pd.read_csv('loan-train.csv')
loan_test = pd.read_csv('loan-test.csv')


# In[5]:


loan_train.head()


# In[6]:


loan_train


# In[7]:


print("Rows: ", len(loan_train))


# In[8]:


print("Columns: ", len(loan_train.columns))


# In[9]:


print("Shape : ", loan_train.shape)


# In[10]:


loan_train_columns = loan_train.columns 
loan_train_columns


# In[11]:


loan_train.describe()


# In[12]:


loan_train.info()


# In[13]:


def explore_object_type(df ,feature_name):
    
    if df[feature_name].dtype ==  'object':
        print(df[feature_name].value_counts())


# In[14]:


explore_object_type(loan_train, 'Gender')


# In[15]:


for featureName in loan_train_columns:
    if loan_train[featureName].dtype == 'object':
        print('\n"' + str(featureName) + '\'s" Values with count are :')
        explore_object_type(loan_train, str(featureName))


# In[17]:


loan_train
loan_train.isna().sum()


# In[18]:


loan_train['Credit_History'].fillna(loan_train['Credit_History'].mode(), inplace=True) 
loan_test['Credit_History'].fillna(loan_test['Credit_History'].mode(), inplace=True)

loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].mean(), inplace=True) 
loan_test['LoanAmount'].fillna(loan_test['LoanAmount'].mean(), inplace=True) 


# In[19]:


loan_train.Loan_Status = loan_train.Loan_Status.replace({"Y": 1, "N" : 0})
 
loan_train.Gender = loan_train.Gender.replace({"Male": 1, "Female" : 0})
loan_test.Gender = loan_test.Gender.replace({"Male": 1, "Female" : 0})

loan_train.Married = loan_train.Married.replace({"Yes": 1, "No" : 0})
loan_test.Married = loan_test.Married.replace({"Yes": 1, "No" : 0})

loan_train.Self_Employed = loan_train.Self_Employed.replace({"Yes": 1, "No" : 0})
loan_test.Self_Employed = loan_test.Self_Employed.replace({"Yes": 1, "No" : 0})


# In[20]:


loan_train['Gender'].fillna(loan_train['Gender'].mode()[0], inplace=True)
loan_test['Gender'].fillna(loan_test['Gender'].mode()[0], inplace=True)

loan_train['Dependents'].fillna(loan_train['Dependents'].mode()[0], inplace=True)
loan_test['Dependents'].fillna(loan_test['Dependents'].mode()[0], inplace=True)

loan_train['Married'].fillna(loan_train['Married'].mode()[0], inplace=True)
loan_test['Married'].fillna(loan_test['Married'].mode()[0], inplace=True)

loan_train['Credit_History'].fillna(loan_train['Credit_History'].mean(), inplace=True)
loan_test['Credit_History'].fillna(loan_test['Credit_History'].mean(), inplace=True)


# In[21]:


from sklearn.preprocessing import LabelEncoder
feature_col = ['Property_Area','Education', 'Dependents']
le = LabelEncoder()
for col in feature_col:
    loan_train[col] = le.fit_transform(loan_train[col])
    loan_test[col] = le.fit_transform(loan_test[col])


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('dark')


# In[23]:


loan_train


# In[24]:


loan_train.plot(figsize=(18, 8))

plt.show()


# In[25]:


plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)


loan_train['ApplicantIncome'].hist(bins=10)
plt.title("Loan Application Amount ")

plt.subplot(1, 2, 2)
plt.grid()
plt.hist(np.log(loan_train['LoanAmount']))
plt.title("Log Loan Application Amount ")

plt.show()


# In[26]:


plt.figure(figsize=(18, 6))
plt.title("Relation Between Applicatoin Income vs Loan Amount ")

plt.grid()
plt.scatter(loan_train['ApplicantIncome'] , loan_train['LoanAmount'], c='k', marker='x')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()


# In[28]:


plt.figure(figsize=(12, 6))
plt.plot(loan_train['Loan_Status'], loan_train['LoanAmount'])
plt.title("Loan Application Amount ")
plt.show()


# In[29]:


plt.figure(figsize=(12,8))
sns.heatmap(loan_train.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=.1)
plt.show()


# In[31]:


#lOGISTIC REGRESSION MODEL 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# In[32]:


logistic_model = LogisticRegression()


# In[33]:


#trng prt 
train_features = ['Credit_History', 'Education', 'Gender']

x_train = loan_train[train_features].values
y_train = loan_train['Loan_Status'].values

x_test = loan_test[train_features].values


# In[34]:


logistic_model.fit(x_train, y_train)


# In[35]:


predicted = logistic_model.predict(x_test)


# In[ ]:




