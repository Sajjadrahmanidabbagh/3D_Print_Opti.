# -*- coding: utf-8 -*-
## Code by Sajjad 2021

#connect to drive
from google.colab import drive
drive.mount('/content/drive')

#!pip install Pillow

import PIL
print('Pillow Version:', PIL.__version__)

from PIL import Image,ImageFilter
import numpy as np
import matplotlib.pyplot as plt

path = '/content/drive/MyDrive/ST-Bioprinting/Design 4 Crop/CAD Design/Part #4.JPG'

# Open the image form working directory
image = Image.open(path).resize((200,200)).convert('L')
image = image.filter(ImageFilter.FIND_EDGES)
data = np.array(image)
image = Image.fromarray(data[85:200,:])
image

image = np.array(image)
axis1 = image[:,103]
axis2 = image[53,:]

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(axis1)
axs[0, 1].plot(np.gradient(axis1))
axs[1, 0].plot(axis2)
axs[1, 1].plot(np.gradient(axis2))
axs[0, 0].title.set_text('Axis 1 (90°)')
axs[0, 1].title.set_text('Gradient of Axis 1 (90°)')
axs[1, 0].title.set_text('Axis 2 (180°)')
axs[1, 1].title.set_text('Gradient of Axis 2 (180°)')
fig.suptitle('Design 6', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.85)
plt.show()

zero_crossings1 = np.where(np.gradient(axis1))[0]
zrr1 = (zero_crossings1).size

zero_crossings2 = np.where(np.gradient(axis2))[0]
zrr2 = (zero_crossings2).size

design_complexity = (zrr1+zrr2)/2

X_test

# Commented out IPython magic to ensure Python compatibility.
############## ML Models #######################
# 1) Import the dataset
# x: st, pt, pp id, is, ng, designcomplexity
# y: similarityindex
# There is a special library for statistical modelling: pymc3

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('/content/drive/MyDrive/ST-Bioprinting/Dataset-meta (Sajjad_8.12.21).csv')
cols = ['PT', 'PP','Design Complexity']
X = dataset[cols]
y = dataset.iloc[:,11]
classname = dataset.iloc[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ML Models
# 1) Linear Regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

linear = linear_model.LinearRegression()
linear.fit(X_train, Y_train)

# train fitting
Y_pred_train = linear.predict(X_train)

print('Coefficients:', linear.coef_)
print('Intercept:', linear.intercept_)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_train, Y_pred_train))
print(linear.score(X_train, Y_train))

# test fitting
Y_pred_test = linear.predict(X_test)
print('Coefficients:', linear.coef_)
print('Intercept:', linear.intercept_)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, Y_pred_test))

print(linear.score(X_test, Y_test))

# Commented out IPython magic to ensure Python compatibility.
#2) Bayesian Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

init = [1.0, 1e-3]
model = BayesianRidge(alpha_init=init[0], lambda_init=init[1])
model.fit(X_train, Y_train)
  
# Model making a prediction on test data
prediction = model.predict(X_test)
  
# train fitting
Y_pred_train = model.predict(X_train)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_train, Y_pred_train))
print(model.score(X_train, Y_train))

# test fitting
Y_pred_test = model.predict(X_test)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, Y_pred_test))
print(model.score(X_test, Y_test))

# Commented out IPython magic to ensure Python compatibility.
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(criterion='absolute_error')
regressor.fit(X_train, Y_train)

# train fitting
Y_pred_train = regressor.predict(X_train)

print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_train, Y_pred_train))
print(linear.score(X_train, Y_train))

# test fitting
Y_pred_test = regressor.predict(X_test)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, Y_pred_test))

print(regressor.score(X_test, Y_test))

print(Y_pred_test)

from sklearn.model_selection import GridSearchCV
param_grid = {'criterion': ['squared_error', 'absolute_error', 'poisson'],'max_features' : ['auto', 'sqrt', 'log2']}
grid = GridSearchCV(RandomForestRegressor(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)
print(grid.best_estimator_)

# Commented out IPython magic to ensure Python compatibility.
# Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor(criterion='poisson', max_features='auto')
tree.fit(X_train,Y_train)
y_pred = tree.predict(X_test)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, y_pred))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, y_pred))


y_pred

from sklearn.model_selection import GridSearchCV
param_grid = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],'max_features' : ['auto', 'sqrt', 'log2']}
grid = GridSearchCV(DecisionTreeRegressor(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)
print(grid.best_estimator_)

# Commented out IPython magic to ensure Python compatibility.
from sklearn.svm import SVR
regressor = SVR(C=10, degree=2, kernel='sigmoid')
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(Y_test, y_pred))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(Y_test, y_pred))
y_pred

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100],'degree': [2,3,4,5,6],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)
print(grid.best_estimator_)

X.head()

pred = [[10,16,38.5]]
pred = sc.transform(pred)
regressor.predict(pred)

## Visualize Regression

#sns.regplot(x="Design Complexity", y="Similarity_Index", data=dataset);
sns.lmplot(x="PP", y="Similarity_Index_in_Percentage", data=dataset, x_estimator=np.mean)

sns.lmplot(x="PP", y="Similarity_Index", hue="PT", data=dataset, x_estimator=np.mean);

import graphviz 
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, out_file=None) 
graph = graphviz.Source(dot_data)

X.iloc[3, :]

#Leave-one-out Cross-validation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(X)
yset = np.array(y)
y_true, y_pred = list(), list()
model = BayesianRidge(alpha_init=1.0, lambda_init=1e-3)
for i in range(x.shape[0]):
  x_train = np.delete(x, (i), axis=0)
  x_test = x[i,:]
  x_test = x_test.reshape(1,-1)
  y_train = np.delete(yset, (i), axis=0)
  y_test = y[i]
  model.fit(x_train, y_train)
  ypred = model.predict(x_test)
  y_true.append(y_test)
  y_pred.append(ypred)

# Commented out IPython magic to ensure Python compatibility.
print('Mean squared error (MSE): %.2f'
#       % mean_squared_error(y_true, y_pred))
print('Coefficient of determination (R^2): %.2f'
#       % r2_score(y_true, y_pred))

plt.plot(X.iloc[0:10,0],y[0:10])

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(np.array(X[y_kmeans == 0].iloc[:,0]), np.array(X[y_kmeans == 0].iloc[:,1]), s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(np.array(X[y_kmeans == 1].iloc[:,0]), np.array(X[y_kmeans == 1].iloc[:,1]), s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(np.array(X[y_kmeans == 2].iloc[:,0]), np.array(X[y_kmeans == 2].iloc[:,1]), s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(np.array(X[y_kmeans == 3].iloc[:,0]), np.array(X[y_kmeans == 3].iloc[:,1]), s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(np.array(X[y_kmeans == 4].iloc[:,0]), np.array(X[y_kmeans == 4].iloc[:,1]), s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()

np.array(X[y_kmeans == 3].iloc[:,0])

