# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 22:17:34 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
#!pip install sweetviz
# importing sweetviz
import sweetviz as sv
import os
os.chdir('PythonCourse/Final/')

google=pd.read_csv('googleplaystore.csv', nrows=10000)
google.head()

#analyzing the dataset
advert_report = sv.analyze(google)
#display the report
advert_report.show_html('GoogleData.html')

google['Installs']=google['Installs'].str[:-1]
ins_new= google['Installs'].replace(",","", regex=True)
ins_new=pd.to_numeric(ins_new)
google['Installs']=ins_new

price_new= google['Price'].str.replace('$','', regex=True)
price_new=pd.to_numeric(price_new)
google['Price']=price_new


#EXPLORATORY DATA ANALYSIS

google.describe()

google = google.drop_duplicates(subset='App')
google_cat=google.groupby('Category').aggregate({'Reviews':'sum', 'Rating':'mean'})
sorted_data1 = google_cat.sort_values(by='Rating',ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=sorted_data1.index.values, y=sorted_data1['Rating'])
plt.xticks(rotation=80)
plt.xlabel("Category")
plt.ylabel("Rating")
plt.title("Category and Rating")
plt.tight_layout()
plt.show()

#It seems that apps with event category have the highest average rate while dating apps have the lowest.


sorted_data2 = google_cat.sort_values(by='Reviews',ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=sorted_data2.index.values, y=sorted_data2['Reviews'])
plt.xticks(rotation=80)
plt.xlabel("Category")
plt.ylabel("Reviews")
plt.title("Category and Reviews")
plt.tight_layout()
plt.show()

#Data shows that game apps has the highest total reviews, social and communication apps are following it, while events apps has the lowest (despite it has the highest rate)


google_ins=google.groupby('Category').aggregate({'Installs':'sum'})
sorted_data3 = google_ins.sort_values(by='Installs',ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=sorted_data3.index.values, y=sorted_data3['Installs'])
plt.xticks(rotation=80)
plt.xlabel("Category")
plt.ylabel("Installs")
plt.title("Category and Installs")
plt.tight_layout()
plt.show()
#It can be seen from the data frame that game apps has the highest total installments.


sorted_by_reviews = google.sort_values(by=['Reviews'], ascending=False)
plt.figure(figsize=(8,6))
fig = sns.barplot(x=sorted_by_reviews['App'][:20], y=sorted_by_reviews['Reviews'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.tight_layout()
plt.show(fig)

#As can be seen Facebook, WhatsApp and Instagram has the top 3 highest reviews;
# however, they are not in the games category. How this can be possible?
# We saw from Sweetviz parts that games category has the second highest number of apps 
#in this data. Therefore, their sum is greater than communication and social categories.


google_type=google.groupby('Type').aggregate({'Reviews':'sum', 'Rating':'mean'})
google_type

#From the table above, it is clear that free apps have much more higher reviews 
#compared to paid ones (521 fold) as expected. However, paid apps have higher mean rating.

google.corr()

f,ax = plt.subplots(figsize=(8, 6))
sns.heatmap(google.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

#MACHINE LEARNING

columns = ['Rating','Installs','Price']
google.dropna(axis=0, how='any', inplace=True)
google_data=google[columns]
target = ['Reviews']
google_target=google[target]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(google_data, google_target, random_state=42)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

print(model.intercept_)
print(model.coef_)

model.score(Xtest,ytest)

from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model, google_data, google_target, cv=7)
np.median(cv_score)

from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(Xtrain, ytrain) 
y_model = model.predict(Xtest)

model.score(Xtest,ytest)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

from sklearn.model_selection import validation_curve
degree = np.arange(0, 7)
#train_score, val_score = validation_curve(PolynomialRegression(), X, y, 'polynomialfeatures__degree', degree, cv=7)
train_score, val_score = validation_curve(PolynomialRegression(), google_data, google_target, param_name='polynomialfeatures__degree', param_range=degree, cv=7)
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');


from sklearn.model_selection import GridSearchCV
param_grid = {'polynomialfeatures__degree': np.arange(7),
'linearregression__fit_intercept': [True, False],
'linearregression__normalize': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(google_data, google_target)
grid.best_params_

model = grid.best_estimator_

from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model, google_data, google_target, cv=7)
np.median(cv_score)


