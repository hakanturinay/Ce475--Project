import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('Data-Table.csv', sep=';')
df = df.apply(lambda x: x.replace(';', ''))
df = df.dropna()
X = df.drop(["SampleNo","Y"], axis=1)
y = df["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor()
tree_model.fit(train_scaled, y_train)

from sklearn.metrics import r2_score
dt_r2value = r2_score(y_test, tree_model.predict(test_scaled))
print("R-Squared: ", dt_r2value)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

tree_test_mse = mean_squared_error(y_test, tree_model.predict(test_scaled))
tree_test_mae = mean_absolute_error(y_test, tree_model.predict(test_scaled))
print("Decision Tree test mse = ",tree_test_mse," & mae = ",tree_test_mae," & rmse = ", sqrt(tree_test_mse))