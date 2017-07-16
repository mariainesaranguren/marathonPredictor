
# coding: utf-8

# In[ ]:

get_ipython().system('uname -a')


# In[ ]:

import pandas as pd
from marathon_demo import marathon_functions
import numpy as np
import pickle
import xgboost as xgb
from sklearn.svm import SVR, NuSVR
import matplotlib.pyplot as plt
import random
import statistics
import math
get_ipython().magic('matplotlib notebook')

from sklearn.model_selection import train_test_split


# # Building features and pickle data

# In[ ]:

get_ipython().run_cell_magic('time', '', 'def build_features():\n    groups = pickle.load(open("marathon_demo/grouping.p", \'rb\'))\n    Xs = []\n    ys = []\n    \n    for group in groups:\n        if isinstance(group.cat[0], str):\n            male = 1 if \'M\' in group.cat[0] else 0\n        else:\n            male = 1\n        female = abs(1 - male)\n        age = marathon_functions.get_age(group.cat[0])\n        \n        for i in range(0, len(group.distance)):\n            d = group.distance[i]\n            if d == 42000:\n                for j in range(i, len(group.distance)):\n                    dist = group.distance[j]\n                    if dist != 42000:\n                    ys.append(group.total_time[i])                   # Append time to ys \n                    time = (group.total_time[j])\n                    difference = group.year[i] - group.year[j]\n                    features = [dist, time, age, male, female, difference]\n                    Xs.append(features)                             # Append features to xs\n                    \n    pickle.dump(Xs, open(\'xs.pkl\', \'wb\'))                           # Pickle xs (features) into xs.pkl\n    pickle.dump(ys, open(\'ys.pkl\', \'wb\'))                           # Pickle ys (total time) into ys.pkl\n    \n    print("len(Xs): ", len(Xs))\n    print("len(ys): ", len(ys))\n    data = np.asarray(Xs)\n    target = np.asarray(ys)\n    return Xs, ys\n\nXs, ys = build_features()')

