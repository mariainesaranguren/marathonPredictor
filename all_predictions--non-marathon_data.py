
# coding: utf-8

# In[1]:

get_ipython().system('uname -a')


# In[2]:

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import statistics
import math
get_ipython().magic('matplotlib notebook')

from sklearn.model_selection import train_test_split


# # Load Data

# We load the data that was pickled in pickle_data.ipynb
# Pickling and unpickling data means that we only have to form our Xs and ys once and load them every time we want to use them.
# Xs: [dist, time, age, male, female, difference]
# ys: [race_time]

# In[3]:

# Unpickle data
Xs = []
ys = []
Xs, ys = pickle.load(open('xs.pkl', 'rb')), pickle.load(open('ys.pkl', 'rb'))


# In this version, we are using a reduced subset of the data that only includes races that are not marathons. This is because in this case we are particularly interested in comparing our methods with the predictions yielded by the Riegel and Cameron formulas. It does not make sense to use marathon races to predict other marathon race times in the case of the parametric models because those work by relating one distance and its corresponding time to the time of a new distance.

# In[21]:

nonmarathon_Xs, nonmarathon_ys = [], []
for i in range(len(Xs)):
    if Xs[i][0]!=42000:
        nonmarathon_Xs.append(Xs[i])
        nonmarathon_ys.append(ys[i])
Xs = nonmarathon_Xs
ys = nonmarathon_ys
Xs_samp, ys_samp = Xs, ys


# # Hyperparameter optimization

# Hyperparameter optimization is used to find the best parameters for the support vector machine model. The hyperparameters that we solved for are C and gamma. 
# 
# The soft-margin constant, C, is the penalty associated with errors/margin errors. A high value of C means there is a high penalty for errors. Therefore, the size of C and the width of the margin from the decision boundary are inversely related.
# 
# The inverse-width parameter, gamma, relates to the shape of the decision boundary. Smaller values of gamma are more linear and higher values of gamma are more flexible. Too high of a gamma value can lead to overfitting.
# 
# For more info on support vector machines, visit this: http://pyml.sourceforge.net/doc/howto.pdf

# The following image describes the relationship between C and gamma:
#     <img src="hyperparameters.png">

# In[5]:

import itertools
import optunity
import optunity.metrics
import sklearn.svm

# SVM RBF tuning
def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_folds=3)   #  num_iter=2,
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
        model = sklearn.svm.SVR(C=C, gamma=gamma).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[55000, 65000], gamma=[0,1])
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = sklearn.svm.SVR(epsilon=0.01, **optimal_pars).fit(x_train, y_train)     # epsilon = 0.001, 
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)


# We generate training and testing splits to use for our tuning procedure here to find the hyperparameters. These same splits are used later on for cross validation.

# In[6]:

# Generating training and testing splits
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.4, random_state=0)  # 40% for testing, 60% for training


# In[7]:

get_ipython().run_cell_magic('time', '', 'compute_mse_rbf_tuned(X_train, y_train, X_test, y_test)')


# In[8]:

get_ipython().run_cell_magic('time', '', 'print(len(Xs))\nouter_cv = optunity.cross_validated(x=Xs, y=ys, num_folds=3)\n# wrap with outer cross-validation\ncompute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)\ncompute_mse_rbf_tuned()')


# # Prediction Models

# ## Parametric models

# In[10]:

def predict_riegel(Xs):
    # predicted_time = old_time*(new_dist / old_dist)^1.06 
    old_time = [Xs[i][1] for i in range(len(Xs))]
    new_dist = [42200 for i in range(len(Xs))]
    old_dist = [Xs[i][0] for i in range(len(Xs))]
    predicted_time = [old_time[i]*(new_dist[i] / old_dist[i])**1.06 for i in range(len(old_time))]
    return predicted_time


# In[11]:

def cameron(d):
    return 13.49681 - (0.000030363*d) + (835.7114/(d**0.7905))

def cameron_formula(t1, old_dist, new_dist): 
    old_dist_new_dist = float(new_dist)/float(old_dist)
    f_old_dist_new_dist = cameron(old_dist)/cameron(new_dist)
    return t1 * old_dist_new_dist * f_old_dist_new_dist

def predict_cameron(Xs):
# #     a = 13.49681 – 0.048865 * old_dist + 2.438936/(old_dist 0.7905)
# #     b = 13.49681 – 0.048865 * new_dist + 2.438936/(new_dist 0.7905)
# #     predicted_time = (old_time / old_dist) * (a / b) * new_dist 
    predicted_time = []
    old_time = [Xs[i][1] for i in range(len(Xs))]
    new_dist = [42200 for i in range(len(Xs))]
    old_dist = [Xs[i][0] for i in range(len(Xs))]
    for i in range(len(Xs)):
        predicted_time.append((cameron_formula(old_time[i], old_dist[i], new_dist[i])))
    return predicted_time


# ## Machine learning models

# In[12]:

def predict_svr(X_train, y_train, X_test):
    svr_rbf = SVR(kernel='rbf',C=55309.635416666686, epsilon=0.01, gamma=0.007942708333333492)
    svr_rbf.fit(X_train,y_train)
    ys_svr = svr_rbf.predict(X_test)
    return ys_svr


# In[13]:

def predict_xgb(X_train, y_train, X_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    num_round = 10
    params = {'booster': 'gbtree','nthread':7, 'objective': 'reg:linear', 'eta': 0.5, 'max_depth': 7, 'tree_method': 'exact'}
    gbm = xgb.train(dtrain=dtrain, params=params)
    ys_xgb = gbm.predict(xgb.DMatrix(X_test))
    return ys_xgb


# In[14]:

def predict_linear(X_train, y_train, X_test):
    linreg = LinearRegression()
    linreg.fit(X_train,y_train)
    ys_linear = linreg.predict(X_test)
    return ys_linear


# # Model Evaluation

# In[15]:

# All error
def error(pred, ys):
    error = [ys[i]-pred[i] for i in range(len(pred))]
    return error


# In[16]:

# RMSE
def rmse(err):
    total_sqr_err = sum([e**2 for e in err])
    n = len(err)
    return math.sqrt(total_sqr_err/n)


# In[17]:

# Percent Error
def percent_error(err, ys):
    percent_error = [abs(err[i]/ys[i])*100 for i in range(len(ys))]
    mean = statistics.mean(percent_error)
    return percent_error, mean
    


# # Parametric modeling

# ### Riegel 

# In[22]:

get_ipython().run_cell_magic('time', '', 'pred_riegel = predict_riegel(Xs_samp)')


# In[23]:

get_ipython().run_cell_magic('time', '', 'riegel_err = error(pred_riegel, ys_samp)\nriegel_rmse = rmse(riegel_err)\nriegel_pcnt_err_all, riegel_pcnt_err_mean = percent_error(riegel_err, ys_samp)\nprint("Riegel")\nprint("RMSE:", riegel_rmse)\nprint("Mean Percent Error:", riegel_pcnt_err_mean)')


# ### Cameron

# In[24]:

get_ipython().run_cell_magic('time', '', 'pred_cameron = predict_cameron(Xs_samp)\nprint (len(pred_cameron))')


# In[25]:

get_ipython().run_cell_magic('time', '', 'cameron_err = error(pred_cameron, ys_samp)\ncameron_rmse = rmse(cameron_err)\ncameron_pcnt_err_all, cameron_pcnt_err_mean = percent_error(cameron_err, ys_samp)\nprint("Riegel")\nprint("RMSE:", cameron_rmse)\nprint("Mean Percent Error:", cameron_pcnt_err_mean)')


# # Overfit Modeling

# In these models, we use the entire dataset to train the model and use it all again to test the model. By using it all, we are overfitting the models and making them more specific to the particular dataset.

# ### Support Vector Machine Regresion

# In[26]:

get_ipython().run_cell_magic('time', '', "pred_overfit_svr = pickle.load(open('ys_pred_overfit_svr.pkl', 'rb'))      # Unpickle predictions\nprint(len(pred_overfit_svr))\nprint(len(ys))")


# In[27]:

get_ipython().run_cell_magic('time', '', 'overfit_svr_err = error(pred_overfit_svr, ys_samp)\noverfit_svr_rmse = rmse(overfit_svr_err)\noverfit_svr_pcnt_err_all, overfit_svr_pcnt_err_mean = percent_error(overfit_svr_err, ys_samp)\nprint("Overfit Support Vector Machine Regression")\nprint("RMSE:", overfit_svr_rmse)\nprint("Percent Error:", overfit_svr_pcnt_err_mean)')


# The errors calculated here are significantly dependent on the hyperparameters given to the model. Below are the results of several different combinations of C and gamme given by the hyperparameter optimization code.
# 
# svr_rbf = SVR(kernel='rbf',C=61438.638402877594, epsilon=0.01, gamma=0.00010334000102074897)
# 
#          Overfit Support Vector Machine Regression
#          RMSE: 1346.0335344828895
#          Percent Error: 6.26024162293
#          Cross Validated Support Vector Machine Regression
#          RMSE: 1525.3474018912768
#          Percent Error: 7.21158684942
# 
# 
# svr_rbf = SVR(kernel='rbf',C=58149.2838541667, epsilon=0.01, gamma=0.014835759139984472)
#          
#          Overfit Support Vector Machine Regression
#          RMSE: 650.8083819943053
#          Percent Error: 1.74966453973
#          Cross Validated Support Vector Machine Regression
#          RMSE: 2150.248035040479
#          Percent Error: 10.3063876975
# 
# 
# svr_rbf = SVR(kernel='rbf',C=63380.84668368657, epsilon=0.01, gamma=0.0005351685093322039)
#          
#          Overfit Support Vector Machine Regression
#          RMSE: 1240.7442199661846
#          Percent Error: 5.41286422333
#          Cross Validated Support Vector Machine Regression
#          RMSE: 1662.2518423329168
#          Percent Error: 7.67926051696

# ### XGBoost

# In[28]:

get_ipython().run_cell_magic('time', '', 'pred_overfit_xgb = predict_xgb(Xs_samp, ys_samp, Xs_samp)')


# In[29]:

get_ipython().run_cell_magic('time', '', 'overfit_xgb_err = error(pred_overfit_xgb, ys_samp)\noverfit_xgb_rmse = rmse(overfit_xgb_err)\noverfit_xgb_pcnt_err_all, overfit_xgb_pcnt_err_mean = percent_error(overfit_xgb_err, ys_samp)\nprint("Overfit XGB")\nprint("RMSE:", overfit_xgb_rmse)\nprint("Mean Percent Error:", overfit_xgb_pcnt_err_mean)')


# ### Linear Regression

# In[30]:

get_ipython().run_cell_magic('time', '', 'pred_overfit_lin = predict_linear(Xs_samp, ys_samp, Xs_samp)')


# In[31]:

get_ipython().run_cell_magic('time', '', 'overfit_lin_err = error(pred_overfit_lin, ys_samp)\noverfit_lin_rmse = rmse(overfit_lin_err)\noverfit_lin_pcnt_err_all, overfit_lin_pcnt_err_mean = percent_error(overfit_lin_err, ys_samp)\nprint("Overfit Linear Regression")\nprint("RMSE:", overfit_lin_rmse)\nprint("Mean Percent Error:", overfit_lin_pcnt_err_mean)')


# # Cross Validated Modeling

# In these models, we use our training and testing data splits to perform cross validation. These models give us a more realistic idea of how well our models would work with new data (data outside the dataset we have here).

# ### Support Vector Machine Regresion

# In[32]:

get_ipython().run_cell_magic('time', '', 'pred_cv_svr = predict_svr(X_train, y_train, X_test)')


# In[33]:

get_ipython().run_cell_magic('time', '', 'cv_svr_err = error(pred_cv_svr, y_test)\ncv_svr_rmse = rmse(cv_svr_err)\ncv_svr_pcnt_err_all, cv_svr_pcnt_err_mean = percent_error(cv_svr_err, y_test)\nprint("Cross Validated Support Vector Machine Regression")\nprint("RMSE:", cv_svr_rmse)\nprint("Percent Error:", cv_svr_pcnt_err_mean)')


# ### XGBoost

# In[34]:

get_ipython().run_cell_magic('time', '', 'pred_cv_xgb = predict_xgb(X_train, y_train, X_test)')


# In[35]:

get_ipython().run_cell_magic('time', '', 'cv_xgb_err = error(pred_cv_xgb, y_test)\ncv_xgb_rmse = rmse(cv_xgb_err)\ncv_xgb_pcnt_err_all, cv_xgb_pcnt_err_mean = percent_error(cv_xgb_err, y_test)\nprint("Cross Validated XGB")\nprint("RMSE:", cv_xgb_rmse)\nprint("Mean Percent Error:", cv_xgb_pcnt_err_mean)')


# ### Linear Regression

# In[36]:

get_ipython().run_cell_magic('time', '', 'pred_cv_lin = predict_linear(X_train, y_train, X_test)')


# In[37]:

get_ipython().run_cell_magic('time', '', 'cv_lin_err = error(pred_cv_lin, y_test)\ncv_lin_rmse = rmse(cv_lin_err)\ncv_lin_pcnt_err_all, cv_lin_pcnt_err_mean = percent_error(cv_lin_err, y_test)\nprint("Cross Validated Linear Regression")\nprint("RMSE:", cv_lin_rmse)\nprint("Mean Percent Error:", cv_lin_pcnt_err_mean)')


# 
# 
# 
# 
# # Results

# ### Predicted Race Times vs. Actual Race Times

# In[38]:

# All models
w = 10
h = .8
ms,alpha, lw=2, 0.5, 0.5
fig = plt.figure(figsize=(w,w*h))
overfit_svr = plt.scatter(ys_samp, pred_overfit_svr, marker='o', s=2*ms,color='green', lw=lw,alpha=alpha)
overfit_xgb = plt.scatter(ys_samp, pred_overfit_xgb, marker='o', s=2*ms,color='red', lw=lw,alpha=alpha)
overfit_lin = plt.scatter(ys_samp, pred_overfit_lin, marker='o', s=2*ms,color='blue', lw=lw,alpha=alpha)
cv_svr = plt.scatter(y_test, pred_cv_svr, marker='o', s=2*ms,color='yellowgreen', lw=lw,alpha=alpha)
cv_xgb = plt.scatter(y_test, pred_cv_xgb, marker='o', s=2*ms,color='lightcoral', lw=lw,alpha=alpha)
cv_lin = plt.scatter(y_test, pred_cv_lin, marker='o', s=2*ms,color='lightblue', lw=lw,alpha=alpha)
riegel = plt.scatter(ys_samp, pred_riegel, marker='o', s=2*ms,color='black', lw=lw,alpha=alpha)
cameron = plt.scatter(ys_samp, pred_riegel, marker='o', s=2*ms,color='gold', lw=lw,alpha=alpha)


plt.legend((overfit_xgb, cv_xgb, overfit_svr, cv_svr, overfit_lin, cv_lin, riegel, cameron),
           ('Overfit XGB', 'CV XGB', 'Overfit SVR', 'CV SVR', 'Overfit Linear', 'CV Linear', 'Riegel', 'Cameron'),
           scatterpoints=1,
           loc='lower right',
           markerscale = 4,
           ncol=3,
           fontsize=8)
plt.plot([min(y_test),max(y_test)], [min(y_test), max(y_test)], color='black', lw=3*lw,alpha=alpha)
plt.xlabel('Actual Race Times (s)')
plt.ylabel('Predicted Race Times (s)')


# In[39]:

# Only machine learning models
w = 10
h = .8
ms,alpha, lw=2, 0.5, 0.5
fig = plt.figure(figsize=(w,w*h))
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 
overfit_svr = plt.scatter(ys_samp, pred_overfit_svr, marker='o', s=2*ms,color='green', lw=lw,alpha=alpha)
overfit_xgb = plt.scatter(ys_samp, pred_overfit_xgb, marker='o', s=2*ms,color='red', lw=lw,alpha=alpha)
overfit_lin = plt.scatter(ys_samp, pred_overfit_lin, marker='o', s=2*ms,color='blue', lw=lw,alpha=alpha)
cv_svr = plt.scatter(y_test, pred_cv_svr, marker='o', s=2*ms,color='yellowgreen', lw=lw,alpha=alpha)
cv_xgb = plt.scatter(y_test, pred_cv_xgb, marker='o', s=2*ms,color='lightcoral', lw=lw,alpha=alpha)
cv_lin = plt.scatter(y_test, pred_cv_lin, marker='o', s=2*ms,color='lightblue', lw=lw,alpha=alpha)

plt.legend((overfit_xgb, cv_xgb, overfit_svr, cv_svr, overfit_lin, cv_lin),
           ('Overfit XGB', 'CV XGB', 'Overfit SVR', 'CV SVR', 'Overfit Linear', 'CV Linear'),
           scatterpoints=1,
           loc='lower right',
           markerscale = 4,
           ncol=3,
           fontsize=8)
plt.plot([min(y_test),max(y_test)], [min(y_test), max(y_test)], color='black', lw=3*lw,alpha=alpha)
plt.xlabel('Actual Race Times (s)')
plt.ylabel('Predicted Race Times (s)')


# In[40]:

# Only overfit machine learning models
w = 10
h = .8
ms,alpha, lw=2, 0.5, 0.5
fig = plt.figure(figsize=(w,w*h))
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 
overfit_svr = plt.scatter(ys_samp, pred_overfit_svr, marker='o', s=2*ms,color='green', lw=lw,alpha=alpha)
overfit_xgb = plt.scatter(ys_samp, pred_overfit_xgb, marker='o', s=2*ms,color='red', lw=lw,alpha=alpha)
overfit_lin = plt.scatter(ys_samp, pred_overfit_lin, marker='o', s=2*ms,color='blue', lw=lw,alpha=alpha)

plt.legend((overfit_svr, overfit_xgb, overfit_lin),
           ('Overfit SVR', 'Overfit XGB', 'Overfit Linear'),
           scatterpoints=1,
           loc='lower right',
           markerscale = 4,
           ncol=3,
           fontsize=8)
plt.plot([min(y_test),max(y_test)], [min(y_test), max(y_test)], color='black', lw=3*lw,alpha=alpha)
plt.xlabel('Actual Race Times (s)')
plt.ylabel('Predicted Race Times (s)')
# plt.title('Predicted Race Times vs. Actual Race Times')


# In[41]:

# Only cross validated machine learning models
w = 10
h = .8
ms,alpha, lw=2, 0.5, 0.5
fig = plt.figure(figsize=(w,w*h))
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 
cv_svr = plt.scatter(y_test, pred_cv_svr, marker='o', s=2*ms,color='green', lw=lw,alpha=alpha)
cv_xgb = plt.scatter(y_test, pred_cv_xgb, marker='o', s=2*ms,color='red', lw=lw,alpha=alpha)
cv_lin = plt.scatter(y_test, pred_cv_lin, marker='o', s=2*ms,color='blue', lw=lw,alpha=alpha)

plt.legend((cv_xgb, cv_svr, cv_lin),
           ('CV XGB', 'CV SVR', 'CV Linear'),
           scatterpoints=1,
           loc='lower right',
           markerscale = 4,
           ncol=3,
           fontsize=8)
plt.plot([min(y_test),max(y_test)], [min(y_test), max(y_test)], color='black', lw=3*lw,alpha=alpha)
plt.xlabel('Actual Race Times (s)')
plt.ylabel('Predicted Race Times (s)')
# plt.title('Predicted Race Times vs. Actual Race Times')


# Observations:
# - The models work best (are more tightly fit around the 1:1 line) in the mid-speed runner range where there is the most data.
# - Models tend to overpredict time of fast runners and underpredict time for slow runners.

# ### Percent Error vs. Race Times

# In[42]:

def bin_results(percent_error, ys):
    pct_err_binned = []
    actual_times_binned = []
    lower_limit = 0
    upper_limit = 10
    sort_idx = np.argsort(np.array(ys))       # Indices of sorted finish times in decreasing order
    y_sorted = np.array(ys)[sort_idx]         # Sorted by ordered indices
    pct_err_sorted = np.array(percent_error)[sort_idx]   # Sorted by ordered indices
    nbins = 100
    bins = np.linspace(0, len(sort_idx), nbins).astype(int)

    bin_edges = []
    bin_pct_err = []
    for i in range(1, len(bins)):
        lower = bins[i-1]
        upper = bins[i]
        bin_edges.append(y_sorted[lower])
        bin_pct_err.append(np.mean(pct_err_sorted[lower:upper]))
        
    return bin_edges, bin_pct_err


# In[46]:

# All models
w = 10
h = .8
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 

# Overfit SVR
overfit_svr_bin_edges, overfit_svr_bin_pct_err = bin_results(overfit_svr_pcnt_err_all, ys_samp)
overfit_svr_err_pcnt = plt.plot(overfit_svr_bin_edges, overfit_svr_bin_pct_err, '-o', color='green')

# Overfit XGB
overfit_xgb_bin_edges, overfit_xgb_bin_pct_err = bin_results(overfit_xgb_pcnt_err_all, ys_samp)
overfit_xgb_err_pcnt = plt.plot(overfit_xgb_bin_edges, overfit_xgb_bin_pct_err, '-o', color='red')

# CV XGB
cv_xgb_bin_edges, cv_xgb_bin_pct_err = bin_results(cv_xgb_pcnt_err_all, y_test)
cv_xgb_err_pcnt = plt.plot(cv_xgb_bin_edges, cv_xgb_bin_pct_err, '-o', color='lightcoral')

# Overfit Linear
overfit_lin_bin_edges, overfit_lin_bin_pct_err = bin_results(overfit_lin_pcnt_err_all, ys_samp)
overfit_lin_err_pcnt = plt.plot(overfit_lin_bin_edges, overfit_lin_bin_pct_err, '-o', color='blue')
# CV Linear
cv_lin_bin_edges, cv_lin_bin_pct_err = bin_results(cv_lin_pcnt_err_all, y_test)
cv_lin_err_pcnt = plt.plot(cv_lin_bin_edges, cv_lin_bin_pct_err, '-o', color='lightblue')


# Riegel
riegel_bin_edges, riegel_bin_pct_err = bin_results(riegel_pcnt_err_all, ys_samp)
riegel_err_pcnt = plt.plot(riegel_bin_edges, riegel_bin_pct_err, '-o', color='black')

# Cameron
cameron_bin_edges, cameron_bin_pct_err = bin_results(cameron_pcnt_err_all, ys_samp)
cameron_err_pcnt = plt.plot(cameron_bin_edges, cameron_bin_pct_err, '-o', color='gold')

plt.xlabel('Marathon Race Time (s)')
plt.ylabel('Error (%)')


# In[47]:

# Overfit models, Riegel, Cameron
w = 10
h = .8
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 

# Overfit SVR
overfit_svr_bin_edges, overfit_svr_bin_pct_err = bin_results(overfit_svr_pcnt_err_all, ys_samp)
overfit_svr_err_pcnt = plt.plot(overfit_svr_bin_edges, overfit_svr_bin_pct_err, '-o', color='green')


# Overfit XGB
overfit_xgb_bin_edges, overfit_xgb_bin_pct_err = bin_results(overfit_xgb_pcnt_err_all, ys_samp)
overfit_xgb_err_pcnt = plt.plot(overfit_xgb_bin_edges, overfit_xgb_bin_pct_err, '-o', color='red')

# Overfit Linear
overfit_lin_bin_edges, overfit_lin_bin_pct_err = bin_results(overfit_lin_pcnt_err_all, ys_samp)
overfit_lin_err_pcnt = plt.plot(overfit_lin_bin_edges, overfit_lin_bin_pct_err, '-o', color='blue')


# Riegel
riegel_bin_edges, riegel_bin_pct_err = bin_results(riegel_pcnt_err_all, ys_samp)
riegel_err_pcnt = plt.plot(riegel_bin_edges, riegel_bin_pct_err, '-o', color='black')

# Cameron
cameron_bin_edges, cameron_bin_pct_err = bin_results(cameron_pcnt_err_all, ys_samp)
cameron_err_pcnt = plt.plot(cameron_bin_edges, cameron_bin_pct_err, '-o', color='gold')


plt.xlabel('Marathon Race Time (s)')
plt.ylabel('Error (%)')


# Observations:
# - The x-axis of this graph has been matched to the x-axis of the graphs of the previous section for direct comparison.
# - The machine learning models outperform parametric models in the region where most of the runners are. 
# - The error of the parametric models increases as marathon time increases. This makes sense because those models were based off of semi-professional athletes and so are biased to faster, less recreational runners and worse fit around the slower runners.
# - The overfit SVR model has the lowest error. This shows that SVR is great at recalling values it has already seen and suggests that the cross-validated model could see significant improvements if the hyperparameters are further optimized.
# - Linear regression has high errors for fastest runners.

# ### Comparison of Prediction Methods by Mean Percent Error

# In[56]:

mean_err = [overfit_xgb_pcnt_err_mean, cv_xgb_pcnt_err_mean, overfit_svr_pcnt_err_mean, cv_svr_pcnt_err_mean, overfit_lin_pcnt_err_mean, cv_lin_pcnt_err_mean, riegel_pcnt_err_mean, cameron_pcnt_err_mean]
names = ["Overfit XBG", "CV XGB\t", "Overfit SVR", "CV SVR\t", "Overfit Linear", "CV Linear", "Riegel\t", "Cameron"]
print("Mean Percent Error by Prediction Method")
for i in range(len(mean_err)):
    print(names[i], "\t", mean_err[i])


# In[51]:

# All methods
w = 10
h = .8
import numpy as na
from matplotlib.pyplot import *
labels = names
data = mean_err
fig = plt.figure(figsize=(w,w*h))
xlocations = na.array(range(len(data)))+0.5
width = 0.7
bar(xlocations, data, width=width)
xticks(xlocations+ width/8, labels)
xlabel('Prediction Method')
ylabel('Error (%)')


# In[52]:

# Only cross validated
w = 10
h = .8
import numpy as na
from matplotlib.pyplot import *
labels = ["CV XGB", "CV SVR", "CV Linear"]
data =   [cv_xgb_pcnt_err_mean, cv_svr_pcnt_err_mean, cv_lin_pcnt_err_mean]

fig = plt.figure(figsize=(w,w*h))
xlocations = na.array(range(len(data)))+0.5
width = 0.7
bar(xlocations, data, width=width)
xticks(xlocations+ width/5, labels)
xlabel('Prediction Method')
ylabel('Error (%)')


# Observations:
# - In this dataset, the average marathon race was 250.93103838 min (~4 hours 11 min). 1% of this time is 2.5093103838 min. Adding up a couple of these 1% errors are really a significant difference, especially because runners typically tend to have quite specific goals for races. For example, a 4% error in a prediction could lead a runner to be expecting a marathon time 10 minutes off of what it actually could be. This could make the difference of following one pacer or another or running too hard or too slowly and could even lead to an injury, hitting the wall, or not finishing a race.
# -  Machine learning methods show great potential for generating predictions. In this experiment, they have already outperformed the parameteric models. With greater data sets we can only expect these errors to decrease.
# - We should remember that these models are only made of the non-marathon data for direct comparison with the parametric models. When the full dataset is used, these errors are smaller. See all_predictions.ipnyb for those results.
# - SVR has significantly different performance between the overfit model and the cross-validated model. The low error of the overfit model proves that SVR has great recall power but the high error of the overfit model suggests this particular model (with the hyperparameters we gave it) is not well suited for generalizing and predicting new data points. However, it is likely that this error could be reduced if the model were better tuned.

# In[ ]:



