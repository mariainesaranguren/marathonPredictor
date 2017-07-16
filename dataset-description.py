
# coding: utf-8

# # Dataset Description

# The dataset used for this project is a few years worth of the Dublin Marathon Race Series.

# In[3]:

import matplotlib.pyplot as plt


# ## Load Data

# In[ ]:

# Unpickle data
Xs = []
ys = []
Xs, ys = pickle.load(open('xs.pkl', 'rb')), pickle.load(open('ys.pkl', 'rb'))      # Unpickle predictions


# ## Gender Breakdown

# In[150]:

male_count = 0
female_count = 0
for i in range(len(Xs)):
    # Xs: dist, time, age, male, female, difference
    if Xs[i][3]==1:    # Males
        male_count+=1
    elif Xs[i][4]==1:  # Females
        female_count+=1
        
print(female_count)
print(male_count)
print(male_count+female_count)


# In[151]:

total = female + male

labels = "Male", "Female"
colors = ['#1F77B4', 'yellowgreen']
sizes = [male_count, female_count]

# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90)
 
plt.axis('equal')
plt.show()


# ## Age Breakdown

# In[154]:

up_to_25, up_to_35, up_to_40, up_to_45, up_to_50, over_55 = 0, 0, 0, 0, 0, 0

# Xs: dist, time, age, male, female, difference
for i in range(len(Xs)):
    # Xs: dist, time, age, male, female, difference
    if Xs[i][2]==25:    # Up to 25
        up_to_25+=1
    elif Xs[i][2]==35:  # Up to 35
        up_to_35+=1
    elif Xs[i][2]==40:  # Up to 40
        up_to_40+=1
    elif Xs[i][2]==45:  # Up to 45
        up_to_45+=1
    elif Xs[i][2]==50:  # Up to 50
        up_to_50+=1
    elif Xs[i][2]==55:  # Over 55
        over_55+=1
print(up_to_25, up_to_35, up_to_40, up_to_45, up_to_50, over_55)


# In[156]:

w = 10
h = .8
ms,alpha, lw=2, 0.5, 0.5
fig = plt.figure(figsize=(w,w*h))
labels = "Up to 25", "Up to 35", "Up to 40", "Up to 45", "Up to 50", "Over 55"
colors = ['gold', '#FF69B4', 'orange', '#00BFFF', '#FF6347', 'yellowgreen']
sizes = [up_to_25, up_to_35, up_to_40, up_to_45, up_to_50, over_55]

# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90)
 
plt.axis('equal')
plt.show()


# ## Marathon Time Distribution

# In[157]:

marathon_Xs, marathon_ys = [], []
for i in range(len(Xs)):
    if Xs[i][0]==42000:
        marathon_Xs.append(Xs[i])
        marathon_ys.append(ys[i])


# In[159]:

# Marathon time distributions
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

w = 8
h = 0.7
ms,alpha, lw=2, 0.5, 0.5

fig = plt.figure(figsize=(w,w*h))
plt.hist(marathon_ys, bins=30)
plt.xlabel('Marathon Race Time (s)')
plt.ylabel('Number of Runners')
plt.show()


# In[63]:

def find_quartiles(data):
    data = sorted(data)
    q2 = int(len(data)/4)
    q3 = int(len(data)/2)
    q4 = int(3*len(data)/4)
    return data[q2], data[q3], data[q4]


# In[65]:

q2, q3, q4 = find_quartiles(marathon_ys)
print(q2, q3, q4)


# In[161]:

import statistics
print("General Marathon Times Stats:")
print("Mean:", statistics.mean(marathon_ys), "s or ", statistics.mean(marathon_ys)/3600, "hrs")
print("Standard Deviation:", statistics.stdev(marathon_ys), "s or ", statistics.stdev(marathon_ys)/60, "min")


# In[165]:

# By gender:
# dist, time, age, male, female, difference
marathon_ys_male = []
marathon_ys_female = []
for i in range(len(marathon_Xs)):
    if marathon_Xs[i][3]==1:    # Males
        marathon_ys_male.append(marathon_ys[i])
    else:                       # Females
        marathon_ys_female.append(marathon_ys[i])

print("Marathon Times Stats by Gender:")
print("Male:")
print("Mean:", statistics.mean(marathon_ys_male), "s or ", statistics.mean(marathon_ys_male)/3600, "hrs")
print("Standard Deviation:", statistics.stdev(marathon_ys_male), "s or ", statistics.stdev(marathon_ys_male)/60, "min")

print("\nFemale:")
print("Mean:", statistics.mean(marathon_ys_female), "s or ", statistics.mean(marathon_ys_female)/3600, "hrs")
print("Standard Deviation:", statistics.stdev(marathon_ys_female), "s or ", statistics.stdev(marathon_ys_male)/60, "min")


# In[ ]:



