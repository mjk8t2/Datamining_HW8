#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics


# In[2]:


df = pd.read_csv('hw8_data.csv')


# In[3]:


df.head(3)


# In[4]:


isattack = df['is_attack'] 


# In[5]:


isattack.head(3)


# In[6]:


df2 = df.drop(['is_attack'], axis=1)


# In[7]:


df2.head(3)


# In[8]:


df.describe()


# In[10]:


train_y,test_y,train_X,test_X = model_selection.train_test_split(df['is_attack'], 
                                                                 df.drop(['is_attack'], axis=1),
                                                                 test_size = 0.7,
                                                                 random_state = 1)


# In[11]:


train_y.shape


# In[12]:


test_y.shape


# In[13]:


train_X.shape


# In[14]:


test_X.shape


# In[15]:


from sklearn import svm


# In[16]:


svc_linear = svm.SVC(kernel='linear', C=1.0)


# In[17]:


svc_linear.fit(train_X,train_y)


# In[18]:


pred_y1 = svc_linear.predict(test_X)


# In[19]:


test_y.value_counts()


# In[20]:


pd.Series(pred_y1).value_counts()


# In[21]:


print(metrics.confusion_matrix(test_y, pred_y1))


# In[22]:


metrics.accuracy_score(test_y, pred_y1)


# In[23]:


print(metrics.classification_report(test_y, pred_y1))


# In[ ]:




