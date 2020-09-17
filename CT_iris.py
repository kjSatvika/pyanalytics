# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:22:11 2020

@author: kjaya
"""

#Clustering using iris dataset

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns     #enhances graphing features
df = data('iris')
df.head()
