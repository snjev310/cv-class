#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read json training file and convert it into data frame
#file = 'training-and-test/training.json'
def read_data(path):
    column = ['English','Physics','Chemistry','Mathematics','Computer Science','Physical Education',
             'Economics','Biology','Accountancy','Business Studies']
    data = None
    with open(path) as train_file:
        lines = train_file.readlines()
        json_obj = [json.loads(line) for line in lines[1:]]
        data = pd.DataFrame(json_obj)
    return data


# In[3]:


training_data = read_data('training-and-test/training.json')


# In[4]:


print(training_data.shape)
training_data.head()


# In[5]:


training_data.count()


# In[6]:


training_data.info()


# In[7]:


training_data.describe()


# In[8]:


training_data = training_data.sample(frac=1).reset_index(drop=True)


# In[9]:


training_data


# In[10]:


training_data['Mathematics'].isna().sum()
training_data.fillna(0,inplace=True)


# In[11]:


training_data['Physics'].isna().sum()


# In[12]:


target = training_data['Mathematics']


# In[13]:


target = pd.DataFrame(target)
target.head()


# In[14]:


#Drop mathematics and Serial no column from training_data
training_data.drop(['Mathematics','serial'],axis=1,inplace=True)


# In[15]:


training_data.head()


# In[16]:


import seaborn as sns


# In[17]:


sns.distplot(training_data['English'])


# In[19]:


training_data.corr()


# In[21]:


corr_math =[(subject, target['Mathematics'].corr(training_data[subject])) for subject in training_data.columns]


# In[31]:


corr_math


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


clf = LogisticRegression()


# In[41]:


clf.fit(training_data[['Physics','Chemistry','BusinessStudies']],target['Mathematics'])


# In[42]:


clf.score(training_data[['Physics','Chemistry','BusinessStudies']],target['Mathematics'])


# In[ ]:




