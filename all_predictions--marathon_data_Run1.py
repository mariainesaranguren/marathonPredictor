
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
# 
# Pickling and unpickling data means that we only have to form our Xs and ys once and load them every time we want to use them.
# 
# Xs: [dist, time, age, male, female, difference]
# 
# ys: [race_time]

# In[3]:

# Unpickle data
Xs = []
ys = []
Xs, ys = pickle.load(open('xs.pkl', 'rb')), pickle.load(open('ys.pkl', 'rb'))


# We use a randomized subset of the full dataset for our machine learning models. This subset makes the calculations faster and gives us an idea of what the models would be like if the full dataset were used. In this version, we do not compare the models with the parametric Riegel and Cameron formulas because we are using data that includes marathon race data points.

# In[4]:

Xs_samp, ys_samp = [], []
for i in range(len(Xs)):
    if random.randint(1, 100) < 30:
        Xs_samp.append(Xs[i])
        ys_samp.append(ys[i])
print(len(Xs), len(ys))
print(len(Xs_samp), len(ys_samp))


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

# In[7]:

# Generating training and testing splits
X_train, X_test, y_train, y_test = train_test_split(Xs_samp, ys_samp, test_size=0.4, random_state=0)  # 40% for testing, 60% for training


# In[7]:

get_ipython().run_cell_magic('time', '', 'compute_mse_rbf_tuned(X_train, y_train, X_test, y_test)')


# In[ ]:

get_ipython().run_cell_magic('time', '', '# print(len(Xs))\n# outer_cv = optunity.cross_validated(x=Xs, y=ys, num_folds=3)\n# # wrap with outer cross-validation\n# compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)\n# compute_mse_rbf_tuned()')


# In[ ]:

# This was not run as it would take too long


# # Prediction Models

# In[10]:

def predict_svr(X_train, y_train, X_test):
    svr_rbf = SVR(kernel='rbf',C=58418.30575452833, epsilon=0.01, gamma=0.001724889476160979)
    svr_rbf.fit(X_train,y_train)
    ys_svr = svr_rbf.predict(X_test)
    return ys_svr


# In[11]:

def predict_xgb(X_train, y_train, X_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    num_round = 10
    params = {'booster': 'gbtree','nthread':7, 'objective': 'reg:linear', 'eta': 0.5, 'max_depth': 7, 'tree_method': 'exact'}
    gbm = xgb.train(dtrain=dtrain, params=params)
    ys_xgb = gbm.predict(xgb.DMatrix(X_test))
    return ys_xgb


# In[12]:

def predict_linear(X_train, y_train, X_test):
    linreg = LinearRegression()
    linreg.fit(X_train,y_train)
    ys_linear = linreg.predict(X_test)
    return ys_linear


# # Model Evaluation

# In[13]:

# All error
def error(pred, ys):
    error = [ys[i]-pred[i] for i in range(len(pred))]
    return error


# In[14]:

# RMSE
def rmse(err):
    total_sqr_err = sum([e**2 for e in err])
    n = len(err)
    return math.sqrt(total_sqr_err/n)


# In[15]:

# Percent Error
def percent_error(err, ys):
    percent_error = [abs(err[i]/ys[i])*100 for i in range(len(ys))]
    mean = statistics.mean(percent_error)
    return percent_error, mean
    


# # Overfit Modeling

# In these models, we use the entire dataset to train the model and use it all again to test the model. By using it all, we are overfitting the models and making them more specific to the particular dataset.

# ### Support Vector Machine Regresion

# In[16]:

get_ipython().run_cell_magic('time', '', 'pred_overfit_svr = predict_svr(Xs_samp, ys_samp, Xs_samp)\nprint(len(pred_overfit_svr))\nprint(len(ys))')


# \* Notice that the cell above took 16h and 6min to run!
# 
# SVR method used here is not very time efficient

# In[17]:

get_ipython().run_cell_magic('time', '', 'overfit_svr_err = error(pred_overfit_svr, ys_samp)\noverfit_svr_rmse = rmse(overfit_svr_err)\noverfit_svr_pcnt_err_all, overfit_svr_pcnt_err_mean = percent_error(overfit_svr_err, ys_samp)\nprint("Overfit Support Vector Machine Regression")\nprint("RMSE:", overfit_svr_rmse)\nprint("Percent Error:", overfit_svr_pcnt_err_mean)')


# The errors calculated here are significantly dependent on the hyperparameters given to the model.

# ### XGBoost

# In[18]:

get_ipython().run_cell_magic('time', '', 'pred_overfit_xgb = predict_xgb(Xs_samp, ys_samp, Xs_samp)')


# In[19]:

get_ipython().run_cell_magic('time', '', 'overfit_xgb_err = error(pred_overfit_xgb, ys_samp)\noverfit_xgb_rmse = rmse(overfit_xgb_err)\noverfit_xgb_pcnt_err_all, overfit_xgb_pcnt_err_mean = percent_error(overfit_xgb_err, ys_samp)\nprint("Overfit XGB")\nprint("RMSE:", overfit_xgb_rmse)\nprint("Mean Percent Error:", overfit_xgb_pcnt_err_mean)')


# ### Linear Regression

# In[20]:

get_ipython().run_cell_magic('time', '', 'pred_overfit_lin = predict_linear(Xs_samp, ys_samp, Xs_samp)')


# In[21]:

get_ipython().run_cell_magic('time', '', 'overfit_lin_err = error(pred_overfit_lin, ys_samp)\noverfit_lin_rmse = rmse(overfit_lin_err)\noverfit_lin_pcnt_err_all, overfit_lin_pcnt_err_mean = percent_error(overfit_lin_err, ys_samp)\nprint("Overfit Linear Regression")\nprint("RMSE:", overfit_lin_rmse)\nprint("Mean Percent Error:", overfit_lin_pcnt_err_mean)')


# # Cross Validated Modeling

# In these models, we use our training and testing data splits to perform cross validation. These models give us a more realistic idea of how well our models would work with new data (data outside the dataset we have here).

# ### Support Vector Machine Regresion

# In[22]:

get_ipython().run_cell_magic('time', '', 'pred_cv_svr = predict_svr(X_train, y_train, X_test)')


# \* Notice that the cell above took 1h and 32min to run!
# 
# SVR method used here is not very time efficient

# In[24]:

get_ipython().run_cell_magic('time', '', 'cv_svr_err = error(pred_cv_svr, y_test)\ncv_svr_rmse = rmse(cv_svr_err)\ncv_svr_pcnt_err_all, cv_svr_pcnt_err_mean = percent_error(cv_svr_err, y_test)\nprint("Cross Validated Support Vector Machine Regression")\nprint("RMSE:", cv_svr_rmse)\nprint("Percent Error:", cv_svr_pcnt_err_mean)')


# ### XGBoost

# In[25]:

get_ipython().run_cell_magic('time', '', 'pred_cv_xgb = predict_xgb(X_train, y_train, X_test)')


# In[26]:

get_ipython().run_cell_magic('time', '', 'cv_xgb_err = error(pred_cv_xgb, y_test)\ncv_xgb_rmse = rmse(cv_xgb_err)\ncv_xgb_pcnt_err_all, cv_xgb_pcnt_err_mean = percent_error(cv_xgb_err, y_test)\nprint("Cross Validated XGB")\nprint("RMSE:", cv_xgb_rmse)\nprint("Mean Percent Error:", cv_xgb_pcnt_err_mean)')


# ### Linear Regression

# In[27]:

get_ipython().run_cell_magic('time', '', 'pred_cv_lin = predict_linear(X_train, y_train, X_test)')


# In[28]:

get_ipython().run_cell_magic('time', '', 'cv_lin_err = error(pred_cv_lin, y_test)\ncv_lin_rmse = rmse(cv_lin_err)\ncv_lin_pcnt_err_all, cv_lin_pcnt_err_mean = percent_error(cv_lin_err, y_test)\nprint("Cross Validated Linear Regression")\nprint("RMSE:", cv_lin_rmse)\nprint("Mean Percent Error:", cv_lin_pcnt_err_mean)')


# 
# 
# 
# 
# # Results

# ### Predicted Race Times vs. Actual Race Times

# In[44]:

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


# In[31]:

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


# In[32]:

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


# In[33]:

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

# In[34]:

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


# In[37]:

# All models
w = 10
h = .8
fig = plt.figure(figsize=(w,w*h))
plt.xlim(8000, 30000) 

# Overfit SVR
overfit_svr_bin_edges, overfit_svr_bin_pct_err = bin_results(overfit_svr_pcnt_err_all, ys_samp)
overfit_svr_err_pcnt = plt.plot(overfit_svr_bin_edges, overfit_svr_bin_pct_err, '-o', color='green')
# CV SVR
cv_svr_bin_edges, cv_svr_bin_pct_err = bin_results(cv_svr_pcnt_err_all, y_test)
cv_svr_err_pcnt = plt.plot(cv_svr_bin_edges, cv_svr_bin_pct_err, '-o', color='yellowgreen')


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


plt.xlabel('Marathon Race Time (s)')
plt.ylabel('Error (%)')


# In[38]:

# Overfit models
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


plt.xlabel('Marathon Race Time (s)')
plt.ylabel('Error (%)')


# Observations:
# - The x-axis of this graph has been matched to the x-axis of the graphs of the previous section for direct comparison.
# - The overfit SVR model has the lowest error. This shows that SVR is great at recalling values it has already seen and suggests that the cross-validated model could see significant improvements if the hyperparameters are further optimized.
# - Linear regression and cross-validated SVR high errors for fastest runners. Cross-validated SVR also has high error for slow runners.

# ### Comparison of Prediction Methods by Mean Percent Error

# In[40]:

mean_err = [overfit_xgb_pcnt_err_mean, cv_xgb_pcnt_err_mean, overfit_svr_pcnt_err_mean, cv_svr_pcnt_err_mean, overfit_lin_pcnt_err_mean, cv_lin_pcnt_err_mean]
names = ["Overfit XBG", "CV XGB\t", "Overfit SVR", "CV SVR\t", "Overfit Linear", "CV Linear"]
print("Mean Percent Error by Prediction Method")
for i in range(len(mean_err)):
    print(names[i], "\t", mean_err[i])


# In[41]:

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


# In[43]:

# Only overfit
w = 10
h = .8
import numpy as na
from matplotlib.pyplot import *
labels = ["Overfit XGB", "Overfit SVR", "Overfit Linear"]
data =   [overfit_xgb_pcnt_err_mean, overfit_svr_pcnt_err_mean, overfit_lin_pcnt_err_mean]

fig = plt.figure(figsize=(w,w*h))
xlocations = na.array(range(len(data)))+0.5
width = 0.7
bar(xlocations, data, width=width)
xticks(xlocations+ width/8, labels)
xlabel('Prediction Method')
ylabel('Error (%)')


# In[42]:

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
# - In this dataset, the average marathon race was 250.93 min (~4 hours 11 min). 1% of this time is 2.51 min. Adding up a couple of these 1% errors are really a significant difference, especially because runners typically tend to have quite specific goals for races. For example, a 4% error in a prediction could lead a runner to be expecting a marathon time 10 minutes off of what it actually could be. This could make the difference of following one pacer or another or running too hard or too slowly and could even lead to an injury, hitting the wall, or not finishing a race.
# - SVR has significantly different performance between the overfit model and the cross-validated model (overfit has 3.27% less error). The low error of the overfit model proves that SVR has great recall power but the high error of the overfit model suggests this particular model (with the hyperparameters we gave it) is not well suited for generalizing and predicting new data points. However, it is likely that this error could be reduced if the model were better tuned.
# - The time efficiency of these models should also be considered. Though SVR achieves high-performing results, especially in the case of the overfit model, it is significantly slower than the other models.
