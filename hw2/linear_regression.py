import numpy as np
import pandas as pd
from scipy import stats
#bonus
from sklearn.linear_model import LinearRegression


def linear_regression(X, y):
    
    b = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T,y))
    e = y - np.matmul(X,b)
    sigma_sqr = (np.matmul(e.T,e)) / (X.shape[0] - X.shape[1] -1)  
    var_b = sigma_sqr*(np.linalg.inv(np.matmul(X.T,X)))
    SE = np.sqrt(var_b)
    mean, var, std = stats.bayes_mvs(X, alpha = 0.95)
    lin_reg = LinearRegression().fit(X, y)
    lin_coef = lin_reg.coef_

    results = {"Regression Coefficients (by hand)": b,
               "Regression Coefficients (with module)": lin_coef,
              "Standard Error": SE ,
              "95% Credible Interval": mean}
    return results
