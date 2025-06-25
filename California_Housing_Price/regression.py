import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import math
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


from sklearn.datasets import fetch_california_housing

# df_housing = fetch_california_housing(as_frame=True)
# df_housing = df_housing.frame

df_house = pd.read_csv("C:\MACHINE LEARNING\Regression\housing.csv")
# print(df_house.head(5))
# print(df_house.isnull().sum())

# Impute missing values

df_house.total_bedrooms = df_house.total_bedrooms.fillna(df_house.total_bedrooms.mean())
# print(df_house.isnull().sum())

# -----------------------LABEL ENCODER------------------------

le = LabelEncoder()
df_house["ocean_proximity"] = le.fit_transform(df_house["ocean_proximity"])
# print(df_house.head(5))


# ---------------------- STANDARDIZE THE DATA -----------------
names = df_house.columns
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_house)
scaled_df = pd.DataFrame(scaled_df, columns=names)
# print(scaled_df.head(5))

# --------------------- BOX PLOT ------------------------

for column in scaled_df:
    plt.figure()
    sns.boxenplot(x=scaled_df[column])
    # plt.show()

a = scaled_df.drop(['median_house_value'], axis=1) 
x=a
y = scaled_df['median_house_value']

# --------------TRAIN TEST SPLIT ---------------------------

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)

# ---------------MODEL CREATION -----------------------------

linreg = LinearRegression()
linreg.fit(x_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)
y_predict = linreg.predict(x_test)
# print(y_predict)

# ------------------ MODEL EVALUATION -----------------------
print("Mean Squared Error:", mean_squared_error(y_test, y_predict))
print("Root Mean Squared Error:", sqrt(mean_squared_error(y_test, y_predict)))
print("R^2 Score:", r2_score(y_test, y_predict))

# ------------------ RANDOM FOREST REGRESOR ----------------------
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg.fit(x_train,y_train)
y_predict = rfreg.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_predict))
print("Root Mean Squared Error:", sqrt(mean_squared_error(y_test, y_predict)))
print("R^2 Score:", r2_score(y_test, y_predict))


# --------------------------- VISUALIZE ---------------------------

scaled_df.plot(kind='scatter', x='median_income', y='median_house_value')
plt.plot(x_test, y_predict, c='red', linewidth=2)
plt.show()
