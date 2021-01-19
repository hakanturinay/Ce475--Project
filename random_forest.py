import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data-Table.csv', sep=';')

df = df.dropna()
X = df.drop(["SampleNo","Y","x2","x4"], axis=1)
y = df["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=0)



from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=25, random_state=0)
rf_model.fit(X_train, y_train)

from sklearn.metrics import r2_score

rf_r2value = r2_score(y_test, rf_model.predict(X_test))
print("R-Squared: ", rf_r2value)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

rf_mse = mean_squared_error(y_train, rf_model.predict(X_train))
rf_mae = mean_absolute_error(y_train, rf_model.predict(X_train))
print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))

rf_test_mse = mean_squared_error(y_test, rf_model.predict(X_test))
rf_test_mae = mean_absolute_error(y_test, rf_model.predict(X_test))
print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", sqrt(rf_test_mse))



plt.figure(figsize=(5, 7))
ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(rf_model.predict(X_test), hist=False, color="b", label="Predicted Values", ax=ax)
plt.title('Y Prediction')
plt.show()
plt.close()

from sklearn import tree
Tree = rf_model.estimators_[5]
plt.figure(figsize=(25, 15))
tree.plot_tree(Tree, filled=True, rounded=True, fontsize=14);
plt.show()

df_2 = pd.read_csv('Data-Table.csv', sep=';', nrows=20, skiprows=range(1,101))
df_3 = df_2.iloc[:, [1,3,5,6]].values


prediction = rf_model.predict(df_3)
predicted_data = pd.DataFrame(np.insert(df_3, 4, prediction, axis=1))
predicted_data.to_excel('predicted_data2.xlsx')
