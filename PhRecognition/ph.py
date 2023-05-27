#!/usr/bin/env python
# coding: utf-8

# # Ph Recognition

# ## Implementation of Decision Trees

# In[55]:


#import libraries
#This is list of all libraries that use in my projects. Note that I dont necessarily use all of them in every project


# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')
import statistics
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

import warnings
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')


# ## Data Preparation and Visualization

# In[2]:


data = pd.read_csv("ph.csv")


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.isna().sum()


# In[6]:


len(data)


# In[7]:


colors = np.array([data.red, data.green, data.blue]).T
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x = data.blue
y = data.green
z = data.red
ax.scatter(x, y, z, c=colors/255.0, s=30)
ax.set_title("Color distribution")
ax.set_xlabel("Blue")
ax.set_ylabel("Green")
ax.set_zlabel("Red")
plt.show()


# In[11]:


def determine_acidity(data):
    if data['ph'] == 7:
        val = 'neutral'
    elif data['ph'] > 7:
        val = 'base'
    elif data['ph'] < 7:
        val = 'acid'
    return val


# In[12]:


data['result'] = data.apply(determine_acidity, axis=1)


# In[13]:


data.head()


# In[14]:


data.groupby("result").mean()


# In[15]:


plt.figure()
sb.countplot(x='ph', data=data)
plt.show()


# In[16]:


plt.figure()
sb.countplot(x='result', data=data)
plt.show()


# In[17]:


plt.figure(figsize=(5,5))

plt.scatter(data['ph'], data['red'])
plt.xlabel('ph value')
plt.ylabel('Red')
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(data['ph'],data['green'])
plt.xlabel('ph value')
plt.ylabel('Green')
plt.show()


plt.figure(figsize=(5,5))
plt.scatter(data['ph'], data['blue'])
plt.xlabel('ph value')
plt.ylabel('Blue')
plt.show()


# In[18]:


plt.figure()
sb.histplot(data['ph'], bins=56, alpha=.5)
plt.show()


# In[19]:


#we will apply regression models on Ph column and classification models on results column:


# ## Regression

# In[20]:


X = data[["blue", "green", "red"]]
y = data["ph"]
feature_names = X.columns


# In[21]:


#linear regression


# In[22]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    scores = r2_score(y_test, y_pred)
    r_scores.append(scores)

max(r_scores)


# In[23]:


#Decision tree


# In[26]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    scores = r2_score(y_test, y_pred)
    r_scores.append(scores)
    
max(r_scores)


# In[27]:


#plotting a decision tree:


# In[28]:


dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=feature_names,
                                filled=True)

graph = graphviz.Source(dot_data, format="png") 
graph


# In[29]:


#randomforest


# In[30]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    scores = r2_score(y_test, y_pred)
    r_scores.append(scores)
    
pd.DataFrame(r_scores, columns = ["Score"]).sort_values(by=["Score"], ascending=False)


# In[31]:


# best model = RanForest
# evaluating the model:
list(zip(rf.feature_importances_, X))


# In[32]:


feature_importance = pd.DataFrame(list(zip(rf.feature_importances_, X)), columns = ["Score","Color"])
feature_importance


# ## Classification

# In[33]:


X = data[["blue", "green", "red"]]
y = data["result"]


# In[34]:


#logreg


# In[35]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    r_scores.append(scores)

max(r_scores)


# In[36]:


#KNN


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
max(scores)


# In[38]:


k_range = list(range(1, 31))
knn = KNeighborsClassifier()
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[39]:


#RanForest


# In[40]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    r_scores.append(scores)
    
max(r_scores)


# In[41]:


#DecisionTree


# In[42]:


r_range = list(range(1, 31))
r_scores = []
for r in r_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=r)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    r_scores.append(scores)
    
pd.DataFrame(r_scores, columns = ["Score"]).sort_values(by=["Score"], ascending=False)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[44]:


y_prob = dt.predict_proba(X_test)
y_prob


# In[45]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[46]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()
plt.show()


# In[47]:


data["result"].replace(("acid", "base", "neutral"), (-1, 1, 0), inplace=True)


# In[48]:


feature_1, feature_2 = np.meshgrid(
     np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max()),
     np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max())
)

grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
tree = DecisionTreeClassifier().fit(data.iloc[:, :2], data.result)
y_pred = np.reshape(tree.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(
     xx0=feature_1, xx1=feature_2, response=y_pred)

display.plot()

display.ax_.scatter(
    data.iloc[:, 0], data.iloc[:, 1], c=data.result, edgecolor="black"
)

plt.show()


# ## Hyper-parameter tuning

# In[49]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[50]:


grid_search = GridSearchCV(dt, 
                           param_grid=params, 
                           cv=4 , scoring = "accuracy")


# In[51]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[52]:


results = pd.DataFrame(grid_search.cv_results_)
results


# In[53]:


max_score = results.sort_values(by=["mean_test_score"], ascending=False).iloc[0]
pd.DataFrame(max_score)


# In[54]:


best_score = grid_search.best_estimator_
print(classification_report(y_test, best_score.predict(X_test)))

