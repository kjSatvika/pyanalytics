# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:39:33 2020

@author: kjaya
"""

#Topic ---- Case Study - Denco - Manufacturing Firm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%case details
#%%Objective
#Expand Business by encouraging loyal customers to Improve repeated sales
#Maximise revenue from high value parts
#%%Information Required
#Who are the most loyal Customers - Improve repeated sales, Target customers with low sales Volumes
#Which customers contribute the most to their revenue - How do I retain these customers & target incentives
#What part numbers bring in to significant portion of revenue - Maximise revenue from high value parts
#What parts have the highest profit margin - What parts are driving profits & what parts need to build further
#%%%
#see all columns
pd.set_option('display.max_columns',15)
#others - max_rows, width, precision, height, date_dayfirst, date_yearfirst
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format
#read data
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv'
df = pd.read_csv(url)
df

# MOST LOYAL
df1 = df.groupby('custname').count().sort_values(by='region',ascending =0)
df1.iloc[0:,0:1]
most_loyal = df1.iloc[0:,0:1].head()
most_loyal

#LOW SALES
low_vol_cust = df1.iloc[0:,0:1].tail()
low_vol_cust 


#MOST_REV
rev = df.groupby('custname').sum().sort_values(by='revenue',ascending =0)
rev.iloc[0:,1] #title missing
most_rev = rev.iloc[0:,1].head()
most_rev

#MAX REV BASED ON PART NUM
df.groupby('partnum').sum().sort_values(by='revenue',ascending =0).iloc[0:,0:1].head()

np.mean
df.groupby
help(np.mean)
