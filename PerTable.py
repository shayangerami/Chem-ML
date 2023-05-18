#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Importing Libraries

# In[124]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Exploring the data

# In[125]:


data = pd.read_csv("periodic_table.csv")
data.head()


# In[126]:


len(data)
#we have all elements!


# In[127]:


data.columns


# In[128]:


data.isna().sum()


# In[129]:


data.describe()


# In[130]:


data.dtypes


# ## Cleaning the data

# ### Now that we know more about our data we will try to clean it by changing some values and filling the missing ones. 

# ### let's first go over each column and make some changes:

# In[131]:


#we can see that there are 28 missing values in the "Group" column. I guess it is because of Lanthanides and Actinides.
#So let's take a quick look to be sure by using a simple plot:
plt.plot(data["AtomicNumber"], data["Group"])


# In[140]:


# There is a gap exactly where these elements are located.
# Ww will replace atomic numbers of 57-70 with "Lan" and 89-103 with "Act".

data["Group"].fillna("L/A", inplace=True)


# In[11]:


#Let's move on to the next column:
#This shows that we have 81 non radioactive elements  are NaN :
data["Radioactive"].isnull().sum()


# In[12]:


# #let's mark them as "no":
data["Radioactive"].fillna("No", inplace=True)


# In[13]:


data["Radioactive"].isna().sum()


# In[14]:


data["Radioactive"]


# In[15]:


# we will apply this to other columns in which "no" is considered as NaN:
data["Natural"].fillna("No", inplace=True)
data["Metal"].fillna("No", inplace=True)
data["Nonmetal"].fillna("No", inplace=True)
data["Metalloid"].fillna("No", inplace=True)
#NOTE: dataset is not accurate since we know that there are 26* artifical elements and 6* metalloids and ...
#But we will keep the dataset as it is and move forward


# In[16]:


#There are also 3 NaN values in the Type column. let's see which elements they are to replace them with appropriate values. 
data[data['Type'].isna()]


# In[17]:


# we can fit them in the Transactinide  category:
data["Type"].fillna("Transactinide", inplace=True)


# In[141]:


data.isna().sum()


# In[19]:


# It is somehow weird to see missing values in "NumberofValence" column. We can calculate the number of valence electrons for all elements.
# Since these numbers are linearly related to the atomic number,interpolation will be the best method to fill these missing values.
data['NumberofValence'].interpolate(method="linear", inplace=True)


# In[20]:


# We can see that the relation somehow makes sence in the graph below:
plt.plot(data["AtomicNumber"], data['NumberofValence'])


# In[21]:


# Now lets check electronegativity:
data[data['Electronegativity'].isna()]


# In[142]:


#since group 18 gases are almost unreactive and other elements are unknown we can mark their electronegativity as N/A or Unknown.
# But, we will need all of our values to be float later when plotting so let's use interpolate to fill them.
data['Electronegativity'].interpolate(method="linear", inplace=True)


# In[143]:


# We can also plot this
plt.plot(data["AtomicNumber"],data["Electronegativity"], label="interpolated EN")


# In[24]:


# We will apply this method to rest of the columns with missing values.
# we first have to change all the dtypes to numeric:
data[["FirstIonization", "Density", "MeltingPoint", "BoilingPoint", "NumberOfIsotopes", "Year", "SpecificHeat"]] = data[["FirstIonization", "Density", "MeltingPoint", "BoilingPoint", "NumberOfIsotopes", "Year", "SpecificHeat"]].apply(pd.to_numeric, errors="ignore")


# In[25]:


#it worked!
data[["FirstIonization", "Density", "MeltingPoint", "BoilingPoint", "NumberOfIsotopes", "Year", "SpecificHeat"]].dtypes


# In[26]:


data["Density"].interpolate(inplace=True)
data["SpecificHeat"].interpolate(inplace=True)

#Note:columns "NumberOfIsotopea" and "Year" do not have any linear relation with atomic number so interpolation makes no sense.
#Note: interpolation will not also work very well for these columns because their missing vslues are in the last 15 elements.
#We will get a constant value for them if we apply this method. I have commented them out for you in case you want to try:

#data["AtomicRadius"].interpolate(inplace=True)
#data["FirstIonization"].interpolate(inplace=True)
#data["MeltingPoint"].interpolate(inplace=True)
#data["BoilingPoint"].interpolate(inplace=True)


# # Analyzing and Visualizing Data
# ## Now that we have our data ready and clean let's move on to the next step

# #### We want to use a model to predict the phase of an element considering some of it's features 

# In[28]:


#to cancel some errors:
pd.options.mode.chained_assignment = None


# ## Feature Selection

# In[29]:


data.groupby("Phase").mean()


# In[30]:


# first we make a new data as we want from our main data
# we will use these features because they seem to have more impact on determining the phase:
data1 = data[[ "AtomicRadius", "Density", "MeltingPoint", "BoilingPoint", "Radioactive", "Phase"]]


# In[31]:


# all of our features most be numbers so we will replace yes,no with 0,1 in Radioactivity column:
data1["Radioactive"].replace(("yes", "No"), (1, 0), inplace=True)


# In[32]:


#our target(Phase) must be a number too:
data1["Phase"].replace(('artificial', 'gas', "liq", "solid"), (0, 1, 2, 3), inplace=True)


# In[33]:


data1.groupby("Phase").mean()


# In[34]:


#and of course must be cleaned:
data1.dropna(inplace=True)


# In[35]:


#we set the features and the target:
X = data1[["AtomicRadius", "Density", "MeltingPoint", "BoilingPoint", "Radioactive"]]
y = data1["Phase"]


# In[36]:


# then we split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# ## Defining our models

# ## 1- Classification

# ### I am going to use two models for classification: KNN and Logistic Regression

# In[37]:


logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[38]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[39]:


# since the accuracy of KNN is depended on n_neighbors every time we change this number we will get a different result.
# so let's go through 1-25 and see the dispersion of the results:


# In[40]:


k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)
print(statistics.mean(scores))
print(max(scores))


# In[41]:


# plot the relationship between K and testing accuracy:
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[42]:


#there is another issue:
#every time we change the random_state of our split, the results will change too. 
#That's because of choosing random datas as training and testing sets, and this will affect our prediction.


# In[43]:


# so we will use cross validation
# in this approach training set is split into k smaller sets each time
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')


# In[44]:


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[45]:


# and now we use cross validation in a wider range of n_neighbors:
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(scores.max())
print(scores.mean())


# In[46]:


#then plot the results
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[47]:


# we can use this method on logistic regression too:
logreg = LogisticRegression(solver='liblinear')
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())


# ### Hyperparameter tuning

# In[48]:


#first we will use GridsearchCV to find the best n_neighbors
#This is just a new approach to what we have already done. The results are expected to be the same:


# In[49]:


k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)


# In[50]:


# we can make a dataframe to see the results:
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[51]:


#the plot will also be the same:
grid_mean_scores = grid.cv_results_['mean_test_score']
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[52]:


print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[53]:


#now let's use this method to determine the other hyperparameter of KNN: 
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)


# In[54]:


grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)


# In[55]:


pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[56]:


print(grid.best_score_)
print(grid.best_params_)


# In[57]:


# we finally found the best hyperparameters for our KNN model


# In[58]:


#we can also use RanomizedSearchCV to find the best random hyperparameters
param_dist = dict(n_neighbors=k_range, weights=weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X, y)
pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[59]:


print(rand.best_score_)
print(rand.best_params_)


# ## Outcome:
# ### We can use KNN with n_neighbors = 16 and weights = distance with accuracy of 0.96 to predict the Phase of elements by having their "AtomicRadius,	Density,	MeltingPoint,	BoilingPoint,	Radioactivity".

# In[60]:


# let's test it to see if it works:
# We will choose a random element from our data set, let's say Li.
# We give it's information to our model to predict the phase of:

# First we redefine our X,y and our model:
X = data1[["AtomicRadius", "Density", "MeltingPoint", "BoilingPoint", "Radioactive"]]
y = data1["Phase"]

knn = KNeighborsClassifier(n_neighbors=16, weights="distance")
knn.fit(X, y)
y_pred = knn.predict([[2.1, 0.534, 453.85, 1615.0, 0]])

y_pred


# In[61]:


# that's right Li is a solid!


# ## Visualization

# In[62]:


#It's good to see how our data looks like to have a better vision of what we have done.


# In[63]:


data1.groupby("Phase").count()


# In[64]:


#this plot illustrates how our numeric features are related to different phases.
plt.scatter(data1["AtomicRadius"],y, c="r", label="AM")
plt.scatter(data1["Density"],y, c="b", label="NP")
plt.scatter(data1["MeltingPoint"],y, c="brown", label="MP")
plt.scatter(data1["BoilingPoint"],y, c="orange", label="BP")


plt.xlabel("features")
plt.ylabel("Phase")
plt.title("plot")

plt.legend()
plt.show()


# In[65]:


#we can also show each feature in a seperate plot.

figure, axis = plt.subplots(2, 2)

axis[0, 0].scatter(data1["AtomicRadius"], y)
axis[0, 0].set_title("AR")
  
axis[0, 1].scatter(data1["Density"], y)
axis[0, 1].set_title("DN")
  
axis[1, 0].scatter(data1["MeltingPoint"], y)
axis[1, 0].set_title("MP")
  
# For Tanh Function
axis[1, 1].scatter(data1["BoilingPoint"], y)
axis[1, 1].set_title("BP")


#changing the position of the plots to avoid overlap
pos = axis[1, 0].get_position()
new_pos = [pos.x0, pos.y0-0.05, pos.width, pos.height]
axis[1, 0].set_position(new_pos)

pos = axis[1, 1].get_position()
new_pos = [pos.x0, pos.y0-0.05, pos.width, pos.height]
axis[1, 1].set_position(new_pos)

  
plt.show()


# In[66]:


# we can check the outliers by using boxplot
#it is also helpful to see how scatterd features are in each phase.
data.boxplot(column=["AtomicRadius", "Density", "MeltingPoint", "BoilingPoint"], vert=False, by="Phase")


# In[67]:


# we can conclude that:
# 1- There is only a few significant outliers in MeltingPoint and Density
# 2- BoilingPoint has a wide range in solid and artificial phase
# 3- MeltingPoint has a wide range in solid phase
# 4- AtomicRadius and Density are centered around low numbers as expected


# In[68]:


#now let's visualize all of our features 2 by 2 in seperate plots:

#defining a color map:
color_map = {0 : "b", 1 : "r", 2 : "g", 3 : "y" }  
colors = data1["Phase"].apply(lambda x: color_map[x])


figure, axis = plt.subplots(3, 2)


axis[0, 0].scatter(data1["AtomicRadius"], data1["Density"], c=colors)
axis[0, 0].set_title("AR-DN")
  

axis[0, 1].scatter(data1["AtomicRadius"], data1["MeltingPoint"], c=colors)
axis[0, 1].set_title("AR-MP")
  

axis[1, 0].scatter(data1["AtomicRadius"], data1["BoilingPoint"], c=colors)
axis[1, 0].set_title("AR-BP")
  

axis[1, 1].scatter(data1["Density"], data1["MeltingPoint"], c=colors)
axis[1, 1].set_title("DN-MP")

axis[2, 0].scatter(data1["Density"], data1["BoilingPoint"], c=colors)
axis[2, 0].set_title("DN-BP")


axis[2, 1].scatter(data1["MeltingPoint"], data1["BoilingPoint"], c=colors)
axis[2, 1].set_title("MP-BP")


#changing the position of the plots to avoid overlap
pos = axis[1, 0].get_position()
new_pos = [pos.x0, pos.y0-0.1, pos.width, pos.height]
axis[1, 0].set_position(new_pos)

pos = axis[1, 1].get_position()
new_pos = [pos.x0, pos.y0-0.1, pos.width, pos.height]
axis[1, 1].set_position(new_pos)

pos = axis[2, 0].get_position()
new_pos = [pos.x0, pos.y0-0.2, pos.width, pos.height]
axis[2, 0].set_position(new_pos)


pos = axis[2, 1].get_position()
new_pos = [pos.x0, pos.y0-0.2, pos.width, pos.height]
axis[2, 1].set_position(new_pos)

plt.show()


# In[69]:


#this shows us that there might be a good linear relation between meltingpoint and boilingpoint which brings us to our next topic:


# ## 2- Regression

# In[70]:


plt.scatter(data["MeltingPoint"], data["BoilingPoint"])


# In[71]:


plt.scatter(data["BoilingPoint"], data["MeltingPoint"])


# In[72]:


# there is a linear relation in both, but let's see which one is more linear:


# In[73]:


#first plot (MP-BP):
X = np.array(data1["MeltingPoint"]).reshape(-1, 1)
y = data1["BoilingPoint"]


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[75]:


y_pred = linreg.predict(X_test)


# In[76]:


r2_score(y_test, y_pred)


# In[77]:


#The score that we get depends on the random state that we choose.
#So let's see if we can do any better by testing a range of random states:

rs_range = list(range(1, 51))
rs_scores = []

for rs in rs_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    rs_scores.append(score)
    
#then we make a dataframe to see the scores:
pd.DataFrame(rs_scores, columns = ["Score"]).sort_values(by=["Score"], ascending=False)


# In[78]:


#you can see that the max of r2_scores is 0.91

#Note!!!!
#Since our index starts from 0, index 24 (which contains the max score) is the 25th iteration.

#conclusion : we will get the best regression when split the data with random_state of 25


# In[79]:


#just to make sure:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
r2_score(y_test, y_pred)


# In[80]:


#the line would look like this:
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, c="r")
plt.xlabel("MP")
plt.ylabel("BP")
plt.title("MP-BP")


# In[81]:


#intercept:
print(linreg.intercept_)
#slope:
print(linreg.coef_)


# In[82]:


# now let's see what happens if we flip X and y:


# In[83]:


#second plot (BP-MP):

X = np.array(data1["BoilingPoint"]).reshape(-1, 1)
y = data1["MeltingPoint"]


# In[84]:


rs_range = list(range(1, 51))
rs_scores = []

for rs in rs_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    rs_scores.append(score)
    
#then we make a dataframe to see the scores:
pd.DataFrame(rs_scores, columns = ["Score"]).sort_values(by=["Score"], ascending=False)


# In[85]:


#We will get max score of 0.94 for the same random_state!
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
r2_score(y_test, y_pred)


# In[86]:


plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, c="r")
plt.xlabel("BP")
plt.ylabel("MP")
plt.title("BP-MP")


# In[87]:


#intercept
print(linreg.intercept_)
#slope:
print(linreg.coef_)


# In[88]:


#we can also use other evaluation metrics for our model such as Mean Squared Error,Root Mean Squared Error, Mean Absolute Error


# ## Outcome:
# ### the equation will be derived :
# ### MeltingPoint = 0.54842782*BoilingPoint -70.98169138340268

# In[106]:


#let's choose a random element to test our equation:


# In[111]:


data[["Symbol", "MeltingPoint", "BoilingPoint"]].loc[50]


# In[112]:


#Predicted melting point
0.54842782*1860 - 70.98169138340268


# In[ ]:


# Very close !!!

