#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


df = pd.read_csv('hw8_data.csv')


# In[3]:


df2 = df.drop(['is_attack'], axis=1)


# In[4]:


df3 = preprocessing.normalize(df2)


# In[5]:


df4 = pd.DataFrame(df3)


# In[6]:


df4.head(3)


# In[7]:


# dropping categorical and including isattack
X = df[[column for column in df if df.nunique()[column] >= 4 or column == "is_attack"]].astype(float)


# In[8]:


X.info()


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


Y = preprocessing.normalize(X)


# In[ ]:





# In[11]:


df_cluster = pd.DataFrame(Y, columns = X.columns)


# In[12]:


df_cluster.head()


# In[13]:


kmeans = KMeans(n_clusters=3)


# In[14]:


kmeans = kmeans.fit(df_cluster[[column for column in df_cluster]])


# In[15]:


df_cluster = df_cluster.assign(Cluster_K=                                    kmeans.predict(df_cluster[[column for column in df_cluster]]))


# In[17]:


df_cluster.head()


# In[18]:


kmeans.predict(df_cluster[[column for column in df_cluster if column != "Cluster_K"]])


# In[19]:


from mpl_toolkits.mplot3d import Axes3D


# In[20]:


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df_cluster['AIT202'],
           df_cluster['AIT203'],
           df_cluster['is_attack'],
           c=df_cluster['Cluster_K'])
plt.show()


# In[ ]:




