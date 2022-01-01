import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('PythonCourse/hw3/')

data = pd.read_csv('cses4_cut.csv')
data.head()
#It seems that data has 33 columns. Let's see the column names.

data.columns
len(data)
#Data consists of 12451 rows.
#Let's see the unique values in each column.
for i in data.columns:
    print(np.sort(data[i].unique()))
#It seems that there are lots of missing and non-informative data in the dataset according to guidebook.

len(data.loc[data['voted'] == True])
#It seems that 10226 of the voted column is True labeled. This is %82 of the data.
#This situatiın may cause some false predictions for False label.


#Therefore, I will drop missing, refused to answer, and answered as "don't know" rows.
#First, I replace 96 to 0 in education column to represent none education.
data['D2003'] = data['D2003'].replace(96,0)
data = data.loc[data['D2003'] != 97]
data = data.loc[data['D2003'] != 98]
data = data.loc[data['D2003'] != 99]

data = data.loc[data['D2004'] != 7]
data = data.loc[data['D2004'] != 8]
data = data.loc[data['D2004'] != 9]

data = data.loc[data['D2005'] != 7]
data = data.loc[data['D2005'] != 8]
data = data.loc[data['D2005'] != 9]

data = data.loc[data['D2006'] != 7]
data = data.loc[data['D2006'] != 8]
data = data.loc[data['D2006'] != 9]

data = data.loc[data['D2007'] != 7]
data = data.loc[data['D2007'] != 8]
data = data.loc[data['D2007'] != 9]

data = data.loc[data['D2008'] != 7]
data = data.loc[data['D2008'] != 8]
data = data.loc[data['D2008'] != 9]

data = data.loc[data['D2009'] != 7]
data = data.loc[data['D2009'] != 8]
data = data.loc[data['D2009'] != 9]

data = data.loc[data['D2010'] != 97]
data = data.loc[data['D2010'] != 98]
data = data.loc[data['D2010'] != 99]

data = data.loc[data['D2011'] != 996]
data = data.loc[data['D2011'] != 997]
data = data.loc[data['D2011'] != 998]
data = data.loc[data['D2011'] != 999]

data = data.loc[data['D2012'] != 7]
data = data.loc[data['D2012'] != 8]
data = data.loc[data['D2012'] != 9]

data = data.loc[data['D2013'] != 7]
data = data.loc[data['D2013'] != 8]
data = data.loc[data['D2013'] != 9]

data = data.loc[data['D2014'] != 7]
data = data.loc[data['D2014'] != 8]
data = data.loc[data['D2014'] != 9]

data = data.loc[data['D2015'] != 97]
data = data.loc[data['D2015'] != 98]
data = data.loc[data['D2015'] != 99]

data = data.loc[data['D2016'] != 996]
data = data.loc[data['D2016'] != 997]
data = data.loc[data['D2016'] != 998]
data = data.loc[data['D2016'] != 999]

data = data.loc[data['D2017'] != 7]
data = data.loc[data['D2017'] != 8]
data = data.loc[data['D2017'] != 9]

data = data.loc[data['D2018'] != 7]
data = data.loc[data['D2018'] != 8]
data = data.loc[data['D2018'] != 9]

data = data.loc[data['D2019'] != 7]
data = data.loc[data['D2019'] != 8]
data = data.loc[data['D2019'] != 9]

data = data.loc[data['D2020'] != 7]
data = data.loc[data['D2020'] != 8]
data = data.loc[data['D2020'] != 9]

data = data.loc[data['D2021'] != 97]
data = data.loc[data['D2021'] != 98]
data = data.loc[data['D2021'] != 99]

data = data.loc[data['D2022'] != 97]
data = data.loc[data['D2022'] != 98]
data = data.loc[data['D2022'] != 99]

data = data.loc[data['D2023'] != 97]
data = data.loc[data['D2023'] != 98]
data = data.loc[data['D2023'] != 99]

data = data.loc[data['D2024'] != 7]
data = data.loc[data['D2024'] != 8]
data = data.loc[data['D2024'] != 9]

data = data.loc[data['D2025'] != 7]
data = data.loc[data['D2025'] != 8]
data = data.loc[data['D2025'] != 9]

data = data.loc[data['D2026'] != 97]
data = data.loc[data['D2026'] != 98]
data = data.loc[data['D2026'] != 99]

data = data.loc[data['D2027'] != 997]
data = data.loc[data['D2027'] != 998]
data = data.loc[data['D2027'] != 999]

data = data.loc[data['D2028'] != 99]

data = data.loc[data['D2029'] != 997]
data = data.loc[data['D2029'] != 998]
data = data.loc[data['D2029'] != 999]

data = data.loc[data['D2030'] != 997]
data = data.loc[data['D2030'] != 998]
data = data.loc[data['D2030'] != 999]

data = data.loc[data['D2031'] != 7]
data = data.loc[data['D2031'] != 8]
data = data.loc[data['D2031'] != 9]

#Let's see unique values again.
for i in data.columns:
    print(np.sort(data[i].unique()))

len(data)
#After dropping non-informative rows, I see that huge amount of them is dropped, and I left with 257 rows.

len(data.loc[data['voted'] == True])
#Now 231 of the rows have True label on voted column. This is approximately %90 of the data.

#Columns D2004 and D2029 seems to have only one unique value (not informative), so I drop them.
data = data.drop('D2004', axis=1)
data = data.drop('D2029', axis=1)

#Columns D2011 and D2016 seems to have lots of unique values for occupations and create excessive noise in data, so I choose to drop them.
data = data.drop('D2011', axis=1)
data = data.drop('D2016', axis=1)

#Seperate the data from target column.
X = data.iloc[:,1:-1]
Y = data.iloc[:,-1]

#One-hot encoding
#I will use one-hot encoding to represent categorical data which are not seem to be ordered.
categorical_cols = ['D2026','D2027', 'D2028', 'D2030', 'D2031']
X = pd.get_dummies(X, columns = categorical_cols)
X.columns
#By looking at columns, I see that it matches the output that I expected.

#To go with machine learning part, I will split the data into train and test.
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, random_state=0, train_size=0.50, test_size=0.50)

#First, I will try Gaussian Naive Bayes, and calculate its accuracy.
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(Xtrain, ytrain) 
y_model = model.predict(Xtest)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)
#Accuracy calculated as 25%, the result is not satisfying. I will try someting else.
#Let's use a dimensionality reduction technique as PCA and observe the results.
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X)
X_2D = model.transform(X) 

data['PCA1'] = X_2D[:, 0]
data['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='voted', data=data, fit_reg=False);
#From the PCA graph, we cannot clearly seperate Voted column as True or False.

#I will move on to K-Neighbors Classifier with n=1.
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
X1, X2, y1, y2 = train_test_split(X, Y, random_state=0, train_size=0.50, test_size=0.50)
model.fit(X1, y1)
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)
#Calculated accuracy score is 86%! This result is fairly high and satisying. I will use this one.
#Let's see the output on confusion matrix.

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y2, y2_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');
#It seems that when voted column is True, ML algorithm founds it True 110 times and founds as False 10 times.
#When voted is False, ML Algorithm founds it as False in 3 times, and categorizes as True 8 times.
#For False label, the result seems to be somewhat misleading. I will do cross validation.

y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)
#Both are higher than %84.

from sklearn.model_selection import cross_val_score
cross_val_score(model, X, Y, cv=5)
#The lowest accuracy is calculated as %75.

from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, Y, cv=LeaveOneOut())
scores.mean()
#By using Leave One Out technique, scores mean calculated as %83, which is satisfying.

#I believe that the reason why False label is labeled wrong most of the time is because the huge amount of Voted column consists of True label.
#Hence, I may lost some of the valuable information while data cleaning process.
