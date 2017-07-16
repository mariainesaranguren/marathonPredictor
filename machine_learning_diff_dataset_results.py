
# coding: utf-8

# # Results of machine learning models between the two experiments

# Since the machine learning models were evaluated with two different versions of our dataset, this file shows the difference in performance between the two.

# The following prints the results gathered from both experiments to compare them side by side.

# In[1]:

models = ["Overfit XBG", "CV XGB\t", "Overfit SVR", "CV SVR\t", "Overfit Linear", "CV Linear"]
err_nonmarathon = [6.4022632723, 7.0050367494, 3.1384668095, 9.8212464091, 7.44990257991, 7.36073082222]
err_marathon1 = [3.21840025705, 3.553579898, 1.98818136634, 5.26246777533, 3.83243969855, 3.89496158289]
err_marathon2 = [3.16844741198, 3.55819008894, 1.62284236904, 5.96651922979, 3.85563384338, 3.89779947442]


# In[10]:

print("Mean Percentage Error by Prediction Method:")
print("MODEL \t\t NON-MARATHON DATASET \tFULL DATASET")
for i in range(len(models)):
    print(models[i], "\t", err_nonmarathon[i], "\t\t", err_marathon1[i]+err_marathon2[i]/2)


# Observations:
# - The machine learning models can perform better than experiment 1 alone suggests when given a broader dataset.
# - The differences between the first and second run of the experiment that use the entire dataset come from the fact that a random 30% sample of the entire dataset is used to train the models and make predictions. In the case of the SVR, these subsamples were also used to tune the hyperparameters.
