
* Note: Some files of this project were intentionally left out. Without them, it is impossible to correctly run the program. Please contact me if you are interested in seeing more of this project.

# # Building a Better Marathon Race Predictor

# This project is focused on building a marathon race time predictor using machine learning methods to improve upon current state of the art models such as Riegel's Formula and Cameron's Equivalence Equations.

# In order to both directly compare the machine learning methods with the parametric models and to test the broader potential of the machine learning methods, this project was carried out with two versions of our dataset:
# 
# 1) Only non-marathon dataset
# 
# Since the parametric models are based on relationships between distance and time, it does not make sense to use one instance of a marathon race to predict another instance of another marathon race (they are the same distance). Therefore, to evaluate the parametric and machine learning methods together, this experiment uses a reduced subset of the full dataset that only includes races that are not marathons.
# 
# 2) Full dataset
# 
# The previous experiment gives us an idea of how the parametric and machine learning models compare. However, the results of those do not accurately convey the potential of the machine learning models to make predictions. Having a wider and larger dataset improves the performance of the machine learning methods.

# ## Files in this directory and what they are
| FILE                                        | DESCRIPTION                       |
|---------------------------------------------|-----------------------------------|
| dataset-description                         | Description of the full dataset   |
|---------------------------------------------|-----------------------------------|
| pickle_data                                 | Build features and pickle data    |
|                                             | (generates xs.pkl and ys.pkl)     |
|---------------------------------------------|-----------------------------------|
| xs.pkl                                      | Xs pickle file (features)         |
|---------------------------------------------|-----------------------------------|
| ys.pkl                                      | ys pickle file (race times)       |
|---------------------------------------------|-----------------------------------|
| ys_pred_overfit_svr.pkl                     | Pickled SVR predictions for non-  |
|                                             | marathon data experiment          |
|---------------------------------------------|-----------------------------------|
| all_predictions -- non-marathon data        | Experiment comparing parametric   |
|                                             | and machine learning models       |
|---------------------------------------------|-----------------------------------|
| all_predictions -- marathon data, Run #1    | Experiment with only machine      |
|                                             | models and full dataset           |
|---------------------------------------------|-----------------------------------|
| all_predictions -- marathon data, Run #2    | Another run of the previous       |
|                                             | experiment                        |
|---------------------------------------------|-----------------------------------|
| machine_learning_diff_dataset_results       | Brief comparison of results       |
|                                             | between both experiments          |
# ## Motivation

# - Half a million marathon finishers in 2013 in the US alone.
# - Runners are invested in their hobby and often look for ways to improve their performance. Race time predictions can significantly affect a runner's training regime and their performance on the actual race. For example, a low prediction might give a runner a false sense of better speed and might cause the runner to get hurt or hit the wall during the race.
# - Running data is popularly tracked and relatively accessible.
# - Machine learning has great potential to answer questions like that we are doing here.

# ## Parametric Models

# The Riegel and Cameron models are two popularly used today.
# 
# 
# The Riegel model, produced by Peter Riegel and published in 1977, is featured in many of today’s online marathon time predictors including Runner’s World. The Riegel model calculates a race time based on one previous race and accounts for differences in race length as well as increasing fatigue over the course of the race. (2) The following formula gives the Riegel model:
# 
#     predicted_time = old_time*(new_dist / old_dist)^1.06
# 
# The Cameron Time Equivalence model was created by Dave Cameron as a linear regression model that predicts the speed of the goal race based on one previous race and is given in the following equations:
# 
#     a = 13.49681 – 0.048865 * old_dist + 2.438936/(old_dist^0.7905)
#     b = 13.49681 – 0.048865 * new_dist + 2.438936/(new_dist^0.7905)
#     predicted_time = (old_time / old_dist) * (a / b) * new_dist
# 

# ## Machine Learning Models

# Instead of approaching the challenge of predicting times though a statistical lense and deriving an equation to model the observed behaviour, this project seeks to calculate marathon race time predictions by using machine learning techniques. 
# 
# It is important to note that though these methods might more accurate predictions (among other advantages), these machine learning models do not really provide a clear parametric model that provides some sort of insight about the relationship between race times and the features taken into consideration.
# 
# The machine learning models tested in this project are:
# - Linear Regression
# - Gradient Boosted Regression (with XGBoost)
# - Support Vector Machine Regression

# 
# 
# Last edit:
# Maria Ines Aranguren; May 10, 2017
