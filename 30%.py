#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset = pd.read_csv("loan-train.csv")


# In[5]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'],margins=True)


# In[11]:


dataset.boxplot(column='ApplicantIncome')


# In[12]:


dataset['ApplicantIncome'].hist(bins=20)


# In[13]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[15]:


dataset.boxplot(column='ApplicantIncome',by='Education')


# In[16]:


dataset.boxplot(column='LoanAmount')


# In[17]:


dataset['LoanAmount'].hist(bins=20)


# In[18]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[20]:


dataset.isnull().sum()


# In[21]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[22]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[23]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[24]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[25]:


dataset.LoanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log =dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[27]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[31]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[29]:


dataset.isnull().sum()


# In[32]:


dataset['TotalIncome']= dataset['ApplicantIncome']+ dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[33]:


dataset['TotalIncome_log'].hist(bins=20)


# In[34]:


dataset.head()


# In[36]:


X= dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y= dataset.iloc[:,12].values 


# In[37]:


X


# In[38]:


y


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[40]:


print(X_train)


# In[42]:


from sklearn.preprocessing import LabelEncoder
Labelencoder_X =LabelEncoder()


# In[45]:


for i in range(0,5):
    X_train[:,i]=Labelencoder_X.fit_transform(X_train[:,i])


# In[47]:


X_train[:,7]=Labelencoder_X.fit_transform(X_train[:,7])


# In[48]:


X_train


# In[53]:


labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[54]:


y_train


# In[ ]:




