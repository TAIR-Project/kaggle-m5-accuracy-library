#!/usr/bin/env python
# coding: utf-8

# In[1]:


def change_type(entry):
    return entry - 11000


# In[2]:


def change_num(entry):
    flag = int(entry / 100)
    buff = entry - (flag * 100)
    return (flag-1)*53 + buff 


# In[5]:


def change_wm_yr_wk(df):
    df["wm_yr_wk"] = df["wm_yr_wk"].apply(change_type)
    df["wm_yr_wk"] = df["wm_yr_wk"].apply(change_num)
    return df

