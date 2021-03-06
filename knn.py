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

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(train_scaled, y_train)

from sklearn.metrics import r2_score
knn_r2value = r2_score(y_test, model.predict(test_scaled))
print("R-Squared: ", knn_r2value)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse = mean_squared_error(y_train, model.predict(train_scaled))
mae = mean_absolute_error(y_train, model.predict(train_scaled))

from math import sqrt

rmse_val = []
for K in range(20):
    K = K+1
    model = KNeighborsRegressor(n_neighbors = K)
    model.fit(train_scaled, y_train)
    pred = model.predict(test_scaled)
    error = sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
    # print('RMSE value for k= ' , K , 'is:', error)

curve = pd.DataFrame(rmse_val)
curve.plot()

print("KNN mse = ",mse," & KNN mae = ",mae," & KNN rmse = ", sqrt(mse))
test_mse = mean_squared_error(y_test, model.predict(test_scaled))
test_mae = mean_absolute_error(y_test, model.predict(test_scaled))
print("KNN mse = ",test_mse," & KNN mae = ",test_mae," & KNNrmse = ", sqrt(test_mse))
